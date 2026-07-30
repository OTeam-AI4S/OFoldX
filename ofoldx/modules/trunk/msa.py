r"""
Multiple-sequence-alignment trunk blocks.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..attention import TriangleAttention, TriangleMultiplication
from ..backend import LayerNorm
from ..backend import outer_product_mean as dispatch_outer_product_mean
from ..backend import pair_weighted_average as dispatch_pair_weighted_average
from ..backend import resolve_config_backend
from ..dropout import StructuredDropout
from ..functional import pair_mask_from_msa_mask
from ..transition import Transition


class MsaTrunk(nn.Module):
    r"""
    MSA/pair trunk with homogeneous joint blocks and one final pair-only update.

    The trunk linearly projects the raw MSA features and an optional per-token embedding to width
    ``msa_size``, runs ``num_stages - 1`` homogeneous joint update blocks, then applies a
    [`FinalMsaPairUpdate`][ofoldx.modules.trunk.msa.FinalMsaPairUpdate]. ``num_stages=0`` skips both
    stages, while ``num_stages=1`` runs only the final pair update.

    Args:
        msa_size: Width of the projected MSA representation.
        pair_size: Width of the pair representation.
        msa_feature_size: Width of the raw per-residue MSA feature vector.
        token_feature_size: Width of the optional per-token embedding.
        num_stages: Total number of MSA/pair stages, including the final pair-only update.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        num_msa_attention_heads: Number of heads in the MSA pair-weighted averaging.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
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
        msa_feature_size: int = 34,
        token_feature_size: int = 449,
        num_stages: int = 4,
        outer_product_hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_projection = nn.Linear(msa_feature_size, msa_size, bias=False)
        self.token_projection = nn.Linear(token_feature_size, msa_size, bias=False)
        self.num_stages = num_stages
        self.blocks = nn.ModuleList(
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
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                pair_dropout=pair_dropout,
                backend=backend,
            )
            if num_stages
            else None
        )

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
            `[B, N, N, pair_size]`. The final update leaves the MSA untouched.
        """

        msa = self.msa_projection(msa)
        if token_embedding is not None:
            msa = msa + self.token_projection(token_embedding).unsqueeze(-3)
        if pair_mask is None and msa_mask is not None:
            pair_mask = pair_mask_from_msa_mask(msa_mask)
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        if self.final_pair_update is not None:
            pair = self.final_pair_update(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        return msa, pair


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
        backend: Backend selector forwarded to every submodule.
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
            expansion=expansion,
            dropout=msa_dropout,
            backend=backend,
        )
        self.triangle_pair_update = TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
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


class FinalMsaPairUpdate(nn.Module):
    r"""
    Final MSA-stage update that refines only the pair representation.

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
        backend: Backend selector forwarded to every submodule.
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
            triangle_attention_hidden_size=triangle_attention_hidden_size,
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
    ) -> Tensor:
        r"""
        Add the outer-product mean of the MSA to the pair and run the triangle pair update.

        The MSA passes through unchanged, since the last block of the trunk emits no MSA update.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            The refined pair representation `[B, N, N, pair_size]`.
        """

        pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
        return self.triangle_pair_update(pair, pair_mask=pair_mask)


class MsaUpdateBlock(nn.Module):
    r"""
    MSA update block: pair-weighted averaging residual followed by a transition residual.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        num_attention_heads: Number of heads in the MSA pair-weighted averaging.
        expansion: Expansion factor of the transition.
        dropout: Row-wise dropout probability on the pair-weighted averaging residual.
        backend: Backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        num_attention_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.15,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_attention_heads,
            backend=backend,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=-3)
        self.transition = Transition(msa_size, expansion=expansion, backend=backend)

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
        backend: Backend selector forwarded to every block.
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
        backend: str = "auto",
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
                backend=backend,
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
        backend: Backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
            backend=backend,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=-3)
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=False,
            backend=backend,
        )
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.transition = Transition(pair_size, expansion=expansion, backend=backend)

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
        ending_update = self.triangle_attention_ending_node(
            pair.transpose(-3, -2),
            pair_mask=end_pair_mask,
        )
        pair = pair + self.rowwise_dropout(ending_update).transpose(-3, -2)
        return pair


class BoltzTrianglePairUpdateBlock(TrianglePairUpdateBlock):
    r"""
    Pair update that shares one dropout scalar per pair row or column.

    The Boltz-derived trunks broadcast the structured dropout mask over different axes than
    [`TrianglePairUpdateBlock`][ofoldx.modules.trunk.msa.TrianglePairUpdateBlock]: the row
    branches share one scalar per row, and the ending-node branch is dropped column-wise after
    it is transposed back rather than before. The residual order and the pair transition are
    unchanged, so only the dropout structure differs.

    Args:
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle multiplication.
        triangle_attention_hidden_size: Per-head width of each triangle attention.
        num_triangle_attention_heads: Number of triangle attention heads.
        expansion: Expansion factor of the pair transition.
        dropout: Structured dropout probability.
        backend: Attention backend selector.
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=(-2, -1))
        self.columnwise_dropout = StructuredDropout(dropout, broadcast_dims=(-3, -1))

    def triangle_update(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Apply the four triangle residual branches with row- and column-shared dropout.

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
        ending_update = self.triangle_attention_ending_node(
            pair.transpose(-3, -2),
            pair_mask=end_pair_mask,
        ).transpose(-3, -2)
        return pair + self.columnwise_dropout(ending_update)


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
        backend: Backend selector; resolves to ``"oops"`` or ``"eager"``.

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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        if head_dim is None:
            if msa_size % num_attention_heads != 0:
                raise ValueError(
                    f"msa_size ({msa_size}) must be divisible by num_attention_heads ({num_attention_heads})"
                )
            head_dim = msa_size // num_attention_heads

        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        inner_size = num_attention_heads * head_dim
        self.msa_norm = LayerNorm(msa_size, backend=backend)
        self.value_projection = nn.Linear(msa_size, inner_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
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
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
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
            proj_left_weight=self.left_projection.weight,
            proj_right_weight=self.right_projection.weight,
            proj_left_bias=self.left_projection.bias,
            proj_right_bias=self.right_projection.bias,
            proj_out_weight=self.output_projection.weight,
            proj_out_bias=self.output_projection.bias,
            msa_norm_eps=self.msa_norm.eps,
            denom_eps=0.0,
            divide_before_projection=True,
            clamp_denominator=False,
            backend=self.backend,
        )


class PostProjectionOuterProductMean(nn.Module):
    r"""
    Project an MSA outer-product sum, then normalize the projected pair update.

    The affine output projection is applied to the masked outer-product sum before division by
    ``count + eps``. Consequently its output bias is also divided by the valid-row count. This is the
    exact post-projection-normalization equation used by AlphaFold 3-derived MSA trunks and Evoformer.

    Args:
        msa_size: Width of the MSA representation.
        hidden_size: Hidden width of each of the left/right MSA projections.
        pair_size: Width of the produced pair update.
        eps: Epsilon added to the per-pair valid-row count after projection.
        backend: Backend selector; resolves to ``"oops"`` or ``"eager"``.
        projection_bias: Whether the left and right MSA projections include biases.
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
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.eps = eps
        self.msa_norm = LayerNorm(msa_size, backend=backend)
        self.left_projection = nn.Linear(msa_size, hidden_size, bias=projection_bias)
        self.right_projection = nn.Linear(msa_size, hidden_size, bias=projection_bias)
        self.output_projection = nn.Linear(hidden_size * hidden_size, pair_size)

    def forward(self, msa: Tensor, msa_mask: Tensor | None = None) -> Tensor:
        r"""
        Project the masked outer-product sum and divide the projected result by ``count + eps``.

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
            proj_left_weight=self.left_projection.weight,
            proj_right_weight=self.right_projection.weight,
            proj_left_bias=self.left_projection.bias,
            proj_right_bias=self.right_projection.bias,
            proj_out_weight=self.output_projection.weight,
            proj_out_bias=self.output_projection.bias,
            msa_norm_eps=self.msa_norm.eps,
            denom_eps=self.eps,
            divide_before_projection=False,
            clamp_denominator=False,
            backend=self.backend,
        )
