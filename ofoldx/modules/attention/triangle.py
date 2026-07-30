r"""
Triangle update modules for Pairformer blocks.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..backend import LayerNorm, resolve_config_backend
from ..backend import triangle_multiplication as dispatch_triangle_multiplication
from ..backend.attention import triangle_attention as dispatch_triangle_attention
from .multihead import GatedMultiheadAttention


class TriangleAttention(nn.Module):
    r"""
    Triangle self-attention over the pair representation with a pair-derived per-head bias.

    Layer-norms the pair representation, attends along the key axis with a per-head triangle bias
    that is shared across the starting-node (row) axis, and applies a gated update. This is the
    attention half of a Pairformer triangle block; the row/column direction is fixed by how the caller
    orients ``pair`` before the call.

    Args:
        pair_size: Width of the pair-representation channel (query/key/value and output width).
        head_dim: Per-head dimension of the attention.
        num_attention_heads: Number of attention heads.
        output_bias: Whether the attention output projection carries a learnable bias.
        gate_bias: Whether the output-gate projection carries a learnable bias.
        gate_bias_init: Initial value for the gate bias when ``gate_bias`` is set.
        backend: Operator provider; see [ofoldx.modules.backend][]. A bias-free
            module on the ``oops`` backend uses the
            dedicated OOps-owned ``triangle_attention`` implementation.
    """

    def __init__(
        self,
        pair_size: int = 128,
        head_dim: int = 32,
        num_attention_heads: int = 4,
        *,
        output_bias: bool = False,
        gate_bias: bool = False,
        gate_bias_init: float = 0.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = GatedMultiheadAttention(
            pair_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            query_bias=False,
            output_bias=output_bias,
            gate_bias=gate_bias,
            gate_bias_init=gate_bias_init,
            backend=backend,
        )

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Layer-norm the pair, attend along the key axis with a per-head triangle bias, and gate the update.

        The triangle bias ``b_{jk} = Linear(norm(pair)_{jk})`` is a per-head function of the two attended
        columns, broadcast across the starting-node row axis. When ``pair_mask`` is given, masked keys get a
        ``-inf`` additive bias so they are excluded from the softmax.

        Args:
            pair: Pair representation `[..., N, N, pair_size]`.
            pair_mask: Optional boolean keep-mask `[..., N, N]`; ``False`` key positions are excluded.

        Returns:
            Additive attention update `[..., N, N, pair_size]`; the caller owns the outer residual.
        """

        normalized = self.pair_norm(pair)
        # Triangle bias term b_{jk} = Linear(z_{jk}): a per-head function of the two attended columns, shared across
        # the starting-node row/batch axis. Project the pair, then move heads ahead of the (query, key) axes and add
        # a singleton row axis so the same bias broadcasts over every row.
        projected_bias = self.bias_projection(normalized)  # (..., n_query, n_key, num_heads)
        triangle_bias = projected_bias.movedim(-1, -3).unsqueeze(-4)  # (..., 1, num_heads, n_query, n_key)
        return dispatch_triangle_attention(
            normalized,
            triangle_bias,
            pair_mask,
            projection=self.attention.projection,
            backend=self.attention.backend,
        )


class TriangleMultiplication(nn.Module):
    r"""
    Triangle multiplicative update for the pair representation.

    Forms gated left/right factors from the normalized pair, contracts them over a shared node to mix
    pairwise information along triangle edges, then norms, projects back, and gates by the input. It returns
    an additive triangle update; the caller block owns the outer residual. ``outgoing`` selects the contraction
    direction.

    Args:
        pair_size: Width of the input/output pair-representation channel.
        hidden_size: Width of the intermediate left/right factors.
        outgoing: If ``True`` contract as ``ikh,jkh->ijh`` (outgoing edges); otherwise ``kih,kjh->ijh``
            (incoming edges).
        projection_bias: Whether the value, gate, and output projections include biases.
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 128,
        *,
        outgoing: bool,
        projection_bias: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outgoing = outgoing
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.input_norm = LayerNorm(pair_size, backend=backend)
        self.left_projection = nn.Linear(pair_size, hidden_size, bias=projection_bias)
        self.left_gate_projection = nn.Linear(pair_size, hidden_size, bias=projection_bias)
        self.right_projection = nn.Linear(pair_size, hidden_size, bias=projection_bias)
        self.right_gate_projection = nn.Linear(pair_size, hidden_size, bias=projection_bias)
        self.output_norm = LayerNorm(hidden_size, backend=backend)
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=projection_bias)
        self.gate_projection = nn.Linear(pair_size, pair_size, bias=projection_bias)

    def forward(
        self,
        pair: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Compute the gated additive triangle update for the pair representation.

        Layer-norms the input, forms gated left/right factors `[..., N, N, hidden_size]`, contracts them over
        the shared node (``ikh,jkh->ijh`` outgoing, else ``kih,kjh->ijh`` incoming), then norms, projects back
        to ``pair_size`` and gates by the input.

        Args:
            pair: Pair representation `[..., N, N, pair_size]`.
            pair_mask: Optional pair mask `[..., N, N]`; masked factor entries are zeroed before the
                contraction.

        Returns:
            Additive triangle update `[..., N, N, pair_size]`; the caller owns the outer residual.
        """

        return dispatch_triangle_multiplication(
            pair,
            pair_mask,
            outgoing=self.outgoing,
            input_norm_weight=self.input_norm.weight,
            input_norm_bias=self.input_norm.bias,
            left_value_weight=self.left_projection.weight,
            left_gate_weight=self.left_gate_projection.weight,
            right_value_weight=self.right_projection.weight,
            right_gate_weight=self.right_gate_projection.weight,
            output_norm_weight=self.output_norm.weight,
            output_norm_bias=self.output_norm.bias,
            output_weight=self.output_projection.weight,
            output_gate_weight=self.gate_projection.weight,
            input_norm_eps=self.input_norm.eps,
            output_norm_eps=self.output_norm.eps,
            backend=self.backend,
            left_value_bias=self.left_projection.bias,
            left_gate_bias=self.left_gate_projection.bias,
            right_value_bias=self.right_projection.bias,
            right_gate_bias=self.right_gate_projection.bias,
            output_bias=self.output_projection.bias,
            output_gate_bias=self.gate_projection.bias,
        )
