r"""
MSA update modules.

The MSA trunk conditions the pair representation on a multiple-sequence-alignment (MSA) axis. Each
block averages MSA values with pair-derived weights, projects the MSA into a pair contribution via the
outer-product mean, and refines the pair representation with the triangle updates. The final block emits
no MSA update, since only the refined pair representation is consumed downstream.
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
from ..functional import pair_mask_from_msa_mask, pair_mask_from_token_mask
from ..transition import RowwiseDropout, Transition


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
        expansion: Feed-forward expansion factor of every transition.
        msa_dropout: Row-wise dropout probability on the MSA residual.
        pair_dropout: Row-wise dropout probability on each pair residual.
        ending_node_transpose_bias: Whether ending-node attention derives its pair bias from the
            pre-transpose pair (the DeepMind AlphaFold 3 convention).
        operator_backend: Operator backend selector forwarded to every submodule.
    Examples:
        >>> import torch
        >>> trunk = MsaTrunk(
        ...     msa_size=8, pair_size=4, msa_feature_size=6, token_feature_size=5,
        ...     num_blocks=2, outer_product_hidden_size=4, num_msa_attention_heads=2,
        ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
        ...     num_triangle_attention_heads=2, expansion=2,
        ... ).eval()
        >>> msa, pair = torch.randn(1, 2, 3, 6), torch.randn(1, 3, 3, 4)
        >>> out_msa, out_pair = trunk(msa, pair, torch.randn(1, 3, 5))
        >>> out_msa.shape, out_pair.shape
        (torch.Size([1, 2, 3, 8]), torch.Size([1, 3, 3, 4]))
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
        ending_node_transpose_bias: bool = False,
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
                ending_node_transpose_bias=ending_node_transpose_bias,
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
                    ending_node_transpose_bias=ending_node_transpose_bias,
                    operator_backend=operator_backend,
                )
            )
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

        Examples:
            >>> import torch
            >>> trunk = MsaTrunk(
            ...     msa_size=8, pair_size=4, msa_feature_size=6, token_feature_size=5,
            ...     num_blocks=1, outer_product_hidden_size=4, num_msa_attention_heads=2,
            ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
            ...     num_triangle_attention_heads=2, expansion=2,
            ... ).eval()
            >>> out_msa, out_pair = trunk(torch.randn(1, 2, 3, 6), torch.randn(1, 3, 3, 4))
            >>> out_msa.shape, out_pair.shape
            (torch.Size([1, 2, 3, 8]), torch.Size([1, 3, 3, 4]))
        """
        msa = self.msa_projection(msa)
        if token_embedding is not None:
            msa = msa + self.token_projection(token_embedding).unsqueeze(-3)
        if pair_mask is None and msa_mask is not None:
            pair_mask = pair_mask_from_msa_mask(msa_mask)
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        return msa, pair


class MsaModule(nn.Module):
    r"""
    MSA trunk that builds the MSA state from raw features and refines the pair representation.

    Unlike [`MsaTrunk`][ofoldx.modules.trunk.msa.MsaTrunk], this module builds the MSA state inline
    from integer token ids and deletion features, adds the projected single input embedding, and returns
    only the refined pair representation. The one-hot vocabulary plus the deletion and paired features must
    match ``msa_feature_size`` (``vocab_size + 3`` when ``use_paired_feature`` else ``vocab_size + 2``).

    Args:
        single_size: Width of the single input embedding.
        pair_size: Width of the pair representation.
        msa_size: Width of the projected MSA state.
        msa_feature_size: Expected width of the concatenated MSA feature vector.
        num_blocks: Number of [`MsaLayer`][ofoldx.modules.trunk.msa.MsaLayer] blocks.
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

    Examples:
        >>> import torch
        >>> module = MsaModule(
        ...     single_size=8, pair_size=4, msa_size=8, msa_feature_size=36, num_blocks=1,
        ...     vocab_size=33, outer_product_hidden_size=4, num_msa_attention_heads=2,
        ...     msa_head_dim=4, triangle_attention_hidden_size=2, num_triangle_attention_heads=2,
        ...     expansion=2,
        ... ).eval()
        >>> B, S, N = 1, 2, 3
        >>> zeros = torch.zeros(B, S, N)
        >>> pair = module(
        ...     torch.randn(B, N, N, 4), torch.randn(B, N, 8),
        ...     msa=torch.randint(0, 33, (B, S, N)),
        ...     has_deletion=zeros, deletion_value=zeros, is_paired=zeros,
        ...     msa_mask=torch.ones(B, S, N), token_mask=torch.ones(B, N),
        ... )
        >>> pair.shape
        torch.Size([1, 3, 3, 4])
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
        self.layers = nn.ModuleList(
            MsaLayer(
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
        msa: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        is_paired: Tensor,
        msa_mask: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        r"""
        Build the MSA state from raw features, then refine the pair representation through every layer.

        Concatenates the one-hot MSA token ids (``vocab_size`` classes) with ``has_deletion`` and
        ``deletion_value`` (and ``is_paired`` when ``use_paired_feature``), projects the result to width
        ``msa_size``, and adds the projected single inputs broadcast over the sequence axis. The pair mask
        is derived from ``token_mask``.

        Args:
            pair: Pair representation `[B, N, N, pair_size]`.
            single_inputs: Per-token single input embedding `[B, N, single_size]`.
            msa: Integer MSA token ids `[B, S, N]`.
            has_deletion: Binary deletion-presence feature `[B, S, N]`.
            deletion_value: Continuous deletion-count feature `[B, S, N]`.
            is_paired: Binary paired-sequence feature `[B, S, N]`; used only when
                ``use_paired_feature`` is set.
            msa_mask: MSA validity mask `[B, S, N]`.
            token_mask: Per-token validity mask `[B, N]`.

        Returns:
            The refined pair representation `[B, N, N, pair_size]`; the internal MSA state is not returned.

        Examples:
            >>> import torch
            >>> module = MsaModule(
            ...     single_size=8, pair_size=4, msa_size=8, msa_feature_size=36, num_blocks=1,
            ...     vocab_size=33, outer_product_hidden_size=4, num_msa_attention_heads=2,
            ...     msa_head_dim=4, triangle_attention_hidden_size=2, num_triangle_attention_heads=2,
            ...     expansion=2,
            ... ).eval()
            >>> B, S, N = 1, 2, 3
            >>> zeros = torch.zeros(B, S, N)
            >>> pair = module(
            ...     torch.randn(B, N, N, 4), torch.randn(B, N, 8),
            ...     msa=torch.randint(0, 33, (B, S, N)),
            ...     has_deletion=zeros, deletion_value=zeros, is_paired=zeros,
            ...     msa_mask=torch.ones(B, S, N), token_mask=torch.ones(B, N),
            ... )
            >>> pair.shape
            torch.Size([1, 3, 3, 4])
        """
        features = [
            F.one_hot(msa, num_classes=self.vocab_size).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa_state = self.msa_projection(torch.cat(features, dim=-1))
        msa_state = msa_state + self.token_projection(single_inputs).unsqueeze(1)

        pair_mask = pair_mask_from_token_mask(token_mask)
        for layer in self.layers:
            pair, msa_state = layer(pair, msa_state, pair_mask, msa_mask)
        return pair


class MsaLayer(nn.Module):
    r"""
    One MSA layer: pair-weighted averaging, MSA transition, outer-product mean, then a triangle pair update.

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

    Examples:
        >>> import torch
        >>> layer = MsaLayer(
        ...     msa_size=8, pair_size=4, outer_product_hidden_size=4, num_msa_attention_heads=2,
        ...     msa_head_dim=4, triangle_attention_hidden_size=2, num_triangle_attention_heads=2,
        ...     expansion=2,
        ... ).eval()
        >>> pair, msa = torch.randn(1, 3, 3, 4), torch.randn(1, 2, 3, 8)
        >>> out_pair, out_msa = layer(pair, msa, torch.ones(1, 3, 3), torch.ones(1, 2, 3))
        >>> out_pair.shape, out_msa.shape
        (torch.Size([1, 3, 3, 4]), torch.Size([1, 2, 3, 8]))
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

        Examples:
            >>> import torch
            >>> layer = MsaLayer(
            ...     msa_size=8, pair_size=4, outer_product_hidden_size=4, num_msa_attention_heads=2,
            ...     msa_head_dim=4, triangle_attention_hidden_size=2, num_triangle_attention_heads=2,
            ...     expansion=2,
            ... ).eval()
            >>> out_pair, out_msa = layer(
            ...     torch.randn(1, 3, 3, 4), torch.randn(1, 2, 3, 8),
            ...     torch.ones(1, 3, 3), torch.ones(1, 2, 3),
            ... )
            >>> out_pair.shape, out_msa.shape
            (torch.Size([1, 3, 3, 4]), torch.Size([1, 2, 3, 8]))
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
        ending_node_transpose_bias: Whether ending-node attention derives its pair bias from the
            pre-transpose pair (the DeepMind AlphaFold 3 convention).
        operator_backend: Operator backend selector forwarded to every submodule.

    Examples:
        >>> import torch
        >>> block = MsaPairBlock(
        ...     msa_size=8, pair_size=4, outer_product_hidden_size=4, num_msa_attention_heads=2,
        ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
        ...     num_triangle_attention_heads=2, expansion=2,
        ... ).eval()
        >>> msa, pair = block(torch.randn(1, 2, 3, 8), torch.randn(1, 3, 3, 4))
        >>> msa.shape, pair.shape
        (torch.Size([1, 2, 3, 8]), torch.Size([1, 3, 3, 4]))
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
        ending_node_transpose_bias: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            operator_backend=operator_backend,
        )
        self.msa_stack = MsaUpdateBlock(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            expansion=expansion,
            dropout=msa_dropout,
            operator_backend=operator_backend,
        )
        self.pair_stack = TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=pair_dropout,
            ending_node_transpose_bias=ending_node_transpose_bias,
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
        the triangle pair stack.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Updated ``(msa, pair)`` of shapes `[B, S, N, msa_size]` and `[B, N, N, pair_size]`.

        Examples:
            >>> import torch
            >>> block = MsaPairBlock(
            ...     msa_size=8, pair_size=4, outer_product_hidden_size=4, num_msa_attention_heads=2,
            ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
            ...     num_triangle_attention_heads=2, expansion=2,
            ... ).eval()
            >>> msa, pair = block(torch.randn(1, 2, 3, 8), torch.randn(1, 3, 3, 4))
            >>> msa.shape, pair.shape
            (torch.Size([1, 2, 3, 8]), torch.Size([1, 3, 3, 4]))
        """
        pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
        msa = self.msa_stack(msa, pair, pair_mask=pair_mask)
        pair = self.pair_stack(pair, pair_mask=pair_mask)
        return msa, pair


class FinalMsaPairBlock(nn.Module):
    r"""
    Final MSA block that refines the pair representation only, leaving the MSA unchanged.

    Adds the outer-product mean of the MSA to the pair representation and runs the triangle pair stack.
    The MSA passes through untouched, since the last block of the trunk emits no MSA update.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Feed-forward expansion factor of every transition.
        pair_dropout: Row-wise dropout probability on each pair residual.
        ending_node_transpose_bias: Whether ending-node attention derives its pair bias from the
            pre-transpose pair (the DeepMind AlphaFold 3 convention).
        operator_backend: Operator backend selector forwarded to every submodule.

    Examples:
        >>> import torch
        >>> block = FinalMsaPairBlock(
        ...     msa_size=8, pair_size=4, outer_product_hidden_size=4,
        ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
        ...     num_triangle_attention_heads=2, expansion=2,
        ... ).eval()
        >>> msa, pair = block(torch.randn(1, 2, 3, 8), torch.randn(1, 3, 3, 4))
        >>> msa.shape, pair.shape
        (torch.Size([1, 2, 3, 8]), torch.Size([1, 3, 3, 4]))
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
        ending_node_transpose_bias: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            operator_backend=operator_backend,
        )
        self.pair_stack = TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=pair_dropout,
            ending_node_transpose_bias=ending_node_transpose_bias,
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
        Add the outer-product mean of the MSA to the pair and run the triangle pair stack.

        The MSA passes through unchanged, since the last block of the trunk emits no MSA update.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Tuple of the unchanged ``msa`` `[B, S, N, msa_size]` and the refined ``pair``
            `[B, N, N, pair_size]`.

        Examples:
            >>> import torch
            >>> block = FinalMsaPairBlock(
            ...     msa_size=8, pair_size=4, outer_product_hidden_size=4,
            ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
            ...     num_triangle_attention_heads=2, expansion=2,
            ... ).eval()
            >>> msa, pair = block(torch.randn(1, 2, 3, 8), torch.randn(1, 3, 3, 4))
            >>> msa.shape, pair.shape
            (torch.Size([1, 2, 3, 8]), torch.Size([1, 3, 3, 4]))
        """
        pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
        pair = self.pair_stack(pair, pair_mask=pair_mask)
        return msa, pair


class MsaUpdateBlock(nn.Module):
    r"""
    MSA update block: pair-weighted averaging residual followed by a transition residual.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        num_attention_heads: Number of heads in the MSA pair-weighted averaging.
        expansion: Feed-forward expansion factor of the transition.
        dropout: Row-wise dropout probability on the pair-weighted averaging residual.
        operator_backend: Operator backend selector forwarded to every submodule.

    Examples:
        >>> import torch
        >>> block = MsaUpdateBlock(msa_size=8, pair_size=4, num_attention_heads=2, expansion=2).eval()
        >>> block(torch.randn(1, 2, 3, 8), torch.randn(1, 3, 3, 4)).shape
        torch.Size([1, 2, 3, 8])
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
        self.feed_forward = Transition(msa_size, expansion=expansion, operator_backend=operator_backend)

    def forward(self, msa: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Update the MSA with a row-dropout pair-weighted averaging residual, then a transition residual.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            The updated MSA representation `[B, S, N, msa_size]`.

        Examples:
            >>> import torch
            >>> block = MsaUpdateBlock(msa_size=8, pair_size=4, num_attention_heads=2, expansion=2).eval()
            >>> block(torch.randn(1, 2, 3, 8), torch.randn(1, 3, 3, 4)).shape
            torch.Size([1, 2, 3, 8])
        """
        msa = msa + self.rowwise_dropout(self.msa_pair_weighted_averaging(msa, pair, pair_mask=pair_mask))
        return msa + self.feed_forward(msa)


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
        expansion: Feed-forward expansion factor of the transition.
        dropout: Row-wise dropout probability applied to each residual.
        ending_node_transpose_bias: Whether ending-node attention derives its pair bias from the
            pre-transpose pair (the DeepMind AlphaFold 3 convention).
        operator_backend: Operator backend selector forwarded to every submodule.

    Examples:
        >>> import torch
        >>> block = TrianglePairUpdateBlock(
        ...     pair_size=4, triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
        ...     num_triangle_attention_heads=2, expansion=2,
        ... ).eval()
        >>> block(torch.randn(1, 3, 3, 4)).shape
        torch.Size([1, 3, 3, 4])
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        ending_node_transpose_bias: bool = False,
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
            transpose_bias=ending_node_transpose_bias,
            operator_backend=operator_backend,
        )
        self.rowwise_dropout = RowwiseDropout(dropout)
        self.feed_forward = Transition(pair_size, expansion=expansion, operator_backend=operator_backend)

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

        Examples:
            >>> import torch
            >>> block = TrianglePairUpdateBlock(
            ...     pair_size=4, triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
            ...     num_triangle_attention_heads=2, expansion=2,
            ... ).eval()
            >>> block(torch.randn(1, 3, 3, 4)).shape
            torch.Size([1, 3, 3, 4])
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
        return pair + self.feed_forward(pair)


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

    Examples:
        >>> import torch
        >>> module = MsaPairWeightedAveraging(msa_size=8, pair_size=4, num_attention_heads=2).eval()
        >>> module(torch.randn(1, 2, 3, 8), torch.randn(1, 3, 3, 4)).shape
        torch.Size([1, 2, 3, 8])
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
        self.norm_msa = LayerNorm(msa_size)
        self.value_projection = nn.Linear(msa_size, inner_size, bias=False)
        self.norm_pair = LayerNorm(pair_size)
        self.pair_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.gate_projection = nn.Linear(msa_size, inner_size, bias=False)
        self.output_projection = nn.Linear(inner_size, msa_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, msa: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Average MSA values along the query-residue axis with per-head weights softmaxed from the pair
        representation, then sigmoid-gate and project back.

        The ``oops`` backend delegates to the OOps implementation; the ``eager`` path is the explicit
        reference.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]` masking invalid keys.

        Returns:
            The MSA update `[B, S, N, msa_size]`.

        Examples:
            >>> import torch
            >>> module = MsaPairWeightedAveraging(msa_size=8, pair_size=4, num_attention_heads=2).eval()
            >>> module(torch.randn(1, 2, 3, 8), torch.randn(1, 3, 3, 4)).shape
            torch.Size([1, 2, 3, 8])
        """
        return dispatch_pair_weighted_average(
            msa,
            pair,
            pair_mask,
            norm_msa_weight=self.norm_msa.weight,
            norm_msa_bias=self.norm_msa.bias,
            norm_pair_weight=self.norm_pair.weight,
            norm_pair_bias=self.norm_pair.bias,
            value_weight=self.value_projection.weight,
            gate_weight=self.gate_projection.weight,
            pair_logits_weight=self.pair_projection.weight,
            output_weight=self.output_projection.weight,
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            norm_msa_eps=self.norm_msa.eps,
            norm_pair_eps=self.norm_pair.eps,
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

    Examples:
        >>> import torch
        >>> module = OuterProductMean(msa_size=8, hidden_size=4, pair_size=4).eval()
        >>> module(torch.randn(1, 2, 3, 8)).shape
        torch.Size([1, 3, 3, 4])
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
        self.norm_msa = LayerNorm(msa_size)
        self.left_projection = nn.Linear(msa_size, hidden_size, bias=False)
        self.right_projection = nn.Linear(msa_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size * hidden_size, pair_size)

    def forward(self, msa: Tensor, msa_mask: Tensor | None = None) -> Tensor:
        r"""
        Form the mask-weighted mean over sequence rows of the outer product of the left/right MSA
        projections and project it to a pair update.

        ``divide_before_projection`` and ``eps`` set the normalisation as documented on the class. The
        ``oops`` backend delegates to the OOps implementation; the ``eager`` path is the explicit reference.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]` zeroing invalid rows and counting valid pairs.

        Returns:
            The pair contribution `[B, N, N, pair_size]`.

        Examples:
            >>> import torch
            >>> module = OuterProductMean(msa_size=8, hidden_size=4, pair_size=4).eval()
            >>> module(torch.randn(1, 2, 3, 8)).shape
            torch.Size([1, 3, 3, 4])
        """
        return dispatch_outer_product_mean(
            msa,
            msa_mask,
            norm_weight=self.norm_msa.weight,
            norm_bias=self.norm_msa.bias,
            proj_left_weight=self.left_projection.weight,
            proj_right_weight=self.right_projection.weight,
            proj_left_bias=self.left_projection.bias,
            proj_right_bias=self.right_projection.bias,
            proj_out_weight=self.output_projection.weight,
            proj_out_bias=self.output_projection.bias,
            norm_eps=self.norm_msa.eps,
            denom_eps=self.eps,
            divide_before_projection=self.divide_before_projection,
            operator_backend=self.operator_backend,
        )
