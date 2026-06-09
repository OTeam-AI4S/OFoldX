r"""
Triangle update modules for Pairformer blocks.
"""

from __future__ import annotations

import torch
from oops import LayerNorm
from torch import Tensor, nn

from ..backend import resolve_config_operator_backend
from ..backend import triangle_multiplication as dispatch_triangle_multiplication
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
        operator_backend: Backend selector; ``"auto"`` picks ``oops`` when available and falls back
            to the ``sdpa`` reference otherwise.

    Examples:
        >>> import torch
        >>> attention = TriangleAttention(
        ...     pair_size=8, head_dim=4, num_attention_heads=2, operator_backend="sdpa"
        ... )
        >>> _ = torch.manual_seed(0)
        >>> pair = torch.randn(1, 3, 3, 8)
        >>> attention(pair).shape
        torch.Size([1, 3, 3, 8])
    """

    def __init__(
        self,
        pair_size: int = 128,
        head_dim: int = 32,
        num_attention_heads: int = 4,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.norm_pair = LayerNorm(pair_size)
        self.bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = GatedMultiheadAttention(
            pair_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            query_bias=False,
            operator_backend=operator_backend,
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
            Gated pair update `[..., N, N, pair_size]`.

        Examples:
            >>> import torch
            >>> attention = TriangleAttention(
            ...     pair_size=8, head_dim=4, num_attention_heads=2, operator_backend="sdpa"
            ... )
            >>> _ = torch.manual_seed(0)
            >>> pair = torch.randn(1, 3, 3, 8)
            >>> mask = torch.ones(1, 3, 3, dtype=torch.bool)
            >>> attention(pair, mask).shape
            torch.Size([1, 3, 3, 8])
        """

        normalized = self.norm_pair(pair)
        # Triangle bias term b_{jk} = Linear(z_{jk}): a per-head function of the two attended columns, shared across
        # the starting-node row/batch axis. Project the pair, then move heads ahead of the (query, key) axes and add
        # a singleton row axis so the same bias broadcasts over every row.
        projected_bias = self.bias_projection(normalized)  # (..., n_query, n_key, num_heads)
        triangle_bias = projected_bias.movedim(-1, -3).unsqueeze(-4)  # (..., 1, num_heads, n_query, n_key)
        biases: list[Tensor] = [triangle_bias]
        if pair_mask is not None:
            pair_mask = pair_mask.to(device=normalized.device).bool()
            mask_bias = torch.zeros_like(pair_mask, dtype=normalized.dtype)
            mask_bias = mask_bias.masked_fill(~pair_mask, torch.finfo(normalized.dtype).min)
            biases.insert(0, mask_bias[..., None, None, :])  # (..., n_row, 1, 1, n_key)
        return self.attention(normalized, biases=biases)


class TriangleMultiplication(nn.Module):
    r"""
    Triangle multiplicative update for the pair representation.

    Forms gated left/right factors from the normalized pair, contracts them over a shared node to mix
    pairwise information along triangle edges, then norms, projects back, and gates by the input. This is
    the multiplicative half of a Pairformer triangle block; ``outgoing`` selects the contraction direction.

    Args:
        pair_size: Width of the input/output pair-representation channel.
        hidden_size: Width of the intermediate left/right factors.
        outgoing: If ``True`` contract as ``ikh,jkh->ijh`` (outgoing edges); otherwise ``kih,kjh->ijh``
            (incoming edges).
        operator_backend: Backend selector; ``"auto"`` picks ``oops`` when available and falls back
            to the ``eager`` reference otherwise.

    Examples:
        >>> import torch
        >>> update = TriangleMultiplication(
        ...     pair_size=8, hidden_size=8, outgoing=True, operator_backend="eager"
        ... )
        >>> _ = torch.manual_seed(0)
        >>> pair = torch.randn(1, 3, 3, 8)
        >>> update(pair).shape
        torch.Size([1, 3, 3, 8])
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 128,
        *,
        outgoing: bool,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outgoing = outgoing
        self.operator_backend = resolve_config_operator_backend(
            operator_backend, supported=("oops", "eager"), public_default="eager"
        )
        self.norm_input = LayerNorm(pair_size)
        self.norm_output = LayerNorm(hidden_size)
        self.left_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.left_gate_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.right_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.right_gate_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)
        self.gate_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        pair: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Apply the gated triangle multiplicative update to the pair representation.

        Layer-norms the input, forms gated left/right factors `[..., N, N, hidden_size]`, contracts them over
        the shared node (``ikh,jkh->ijh`` outgoing, else ``kih,kjh->ijh`` incoming), then norms, projects back
        to ``pair_size`` and gates by the input.

        Args:
            pair: Pair representation `[..., N, N, pair_size]`.
            pair_mask: Optional pair mask `[..., N, N]`; masked factor entries are zeroed before the
                contraction.

        Returns:
            Updated pair representation `[..., N, N, pair_size]`.

        Raises:
            OOpsUnsupportedError: If ``operator_backend="oops"`` and ``pair`` / ``pair_mask`` are not CUDA
                tensors (the fused ``triangle_multiplication`` kernel is CUDA-only).

        Examples:
            >>> import torch
            >>> update = TriangleMultiplication(
            ...     pair_size=8, hidden_size=8, outgoing=False, operator_backend="eager"
            ... )
            >>> _ = torch.manual_seed(0)
            >>> pair = torch.randn(1, 3, 3, 8)
            >>> mask = torch.ones(1, 3, 3, dtype=torch.bool)
            >>> update(pair, mask).shape
            torch.Size([1, 3, 3, 8])
        """

        return dispatch_triangle_multiplication(
            pair,
            pair_mask,
            outgoing=self.outgoing,
            norm_input_weight=self.norm_input.weight,
            norm_input_bias=self.norm_input.bias,
            left_value_weight=self.left_projection.weight,
            left_gate_weight=self.left_gate_projection.weight,
            right_value_weight=self.right_projection.weight,
            right_gate_weight=self.right_gate_projection.weight,
            norm_output_weight=self.norm_output.weight,
            norm_output_bias=self.norm_output.bias,
            output_weight=self.output_projection.weight,
            output_gate_weight=self.gate_projection.weight,
            eps=self.norm_input.eps,
            operator_backend=self.operator_backend,
        )
