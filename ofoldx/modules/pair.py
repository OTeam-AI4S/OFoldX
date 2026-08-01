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
Pair-representation triangle updates and stacks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from torch import Tensor, is_grad_enabled, nn

from ..compilation import HomogeneousBlockRegions
from .attention import TriangleAttention
from .backend import LayerNorm, resolve_config_backend
from .backend import triangle_multiplication as dispatch_triangle_multiplication
from .dropout import StructuredDropout
from .transition import Transition


class TriangleMultiplication(nn.Module):
    r"""
    Triangle multiplicative update for the pair representation.

    Forms gated left/right factors from the normalized pair, contracts them over a shared node to mix
    pairwise information along triangle edges, then norms, projects back, and gates by the input. It returns
    an additive triangle update; the caller block owns the outer residual. ``direction`` selects the edges
    the contraction runs over; both directions are the same operation on a different node index.

    Args:
        pair_size: Width of the input/output pair-representation channel.
        hidden_size: Width of the intermediate left/right factors.
        direction: ``"outgoing"`` contracts as ``ikh,jkh->ijh`` (edges leaving both nodes), ``"incoming"`` as
            ``kih,kjh->ijh`` (edges entering both nodes).
        projection_bias: Whether the value, gate, and output projections include biases.
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 128,
        *,
        direction: Literal["incoming", "outgoing"],
        projection_bias: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.direction = direction
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
        the shared node (``ikh,jkh->ijh`` outgoing, ``kih,kjh->ijh`` incoming), then norms, projects back
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
            direction=self.direction,
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


class TrianglePairUpdateBlock(nn.Module):
    r"""
    Pair-only triangle update block: triangle multiplication, triangle attention, then a transition.

    Runs outgoing then incoming triangle multiplication, starting- then ending-node triangle attention
    (each with row dropout), and a transition, all as residual updates on the pair representation.

    Args:
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_head_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Expansion factor of the transition.
        dropout: Row-wise dropout probability applied to each residual.
        backend: Backend selector forwarded to every submodule.
        ending_bias_pair_dims: Source pair axes mapped to ending attention's bias query/key axes.
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
        *,
        ending_bias_pair_dims: tuple[int, int] = (-3, -2),
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            direction="outgoing",
            backend=backend,
        )
        self.rowwise_dropout: nn.Module = StructuredDropout(dropout, broadcast_dims=-3)
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            direction="incoming",
            backend=backend,
        )
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_size=triangle_attention_head_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_size=triangle_attention_head_size,
            num_attention_heads=num_triangle_attention_heads,
            bias_pair_dims=ending_bias_pair_dims,
            backend=backend,
        )
        self.transition = Transition(pair_size, expansion=expansion, backend=backend)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Apply the configured triangle residual updates, then the final pair transition.

        [`triangle_update`][ofoldx.modules.pair.TrianglePairUpdateBlock.triangle_update]
        owns the exact triangle branch order, transpositions, and structured dropout.

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


class TrianglePairUpdateStack(HomogeneousBlockRegions, nn.Module):
    r"""
    Apply a homogeneous sequence of pair-only triangle update blocks.

    Args:
        num_blocks: Number of triangle pair-update blocks.
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_head_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Expansion factor of the transition.
        dropout: Row-wise dropout probability applied to each residual.
        backend: Backend selector forwarded to every block.
        block: Pair-update block factory. Omitted uses ``TrianglePairUpdateBlock``.
    """

    def __init__(
        self,
        num_blocks: int,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
        *,
        block: Callable[..., TrianglePairUpdateBlock] = TrianglePairUpdateBlock,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            block(
                pair_size=pair_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_head_size=triangle_attention_head_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                dropout=dropout,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

    _gradient_checkpointing_func: Callable[..., Tensor]

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
            if self.gradient_checkpointing and self.training and is_grad_enabled():
                pair = self._gradient_checkpointing_func(
                    block.__call__,
                    pair,
                    pair_mask=pair_mask,
                )
            else:
                pair = block(pair, pair_mask=pair_mask)
        return pair


class LineDropoutTrianglePairUpdateBlock(TrianglePairUpdateBlock):
    r"""
    Triangle pair update whose dropout removes whole rows and columns of the pair representation.

    Each residual keeps or drops an entire line of the pair grid with one scalar draw: the two
    multiplicative updates and the starting-node attention drop whole rows, and the ending-node
    attention drops whole columns.
    [`TrianglePairUpdateBlock`][ofoldx.modules.pair.TrianglePairUpdateBlock] instead draws one
    value per column and channel and shares it across rows. The two granularities are different
    regularizers rather than two spellings of one, and this is the one the released Boltz Pairformer
    trains with. Evaluation is unaffected.

    Args:
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_head_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Expansion factor of the transition.
        dropout: Probability of dropping a whole row or column from each residual.
        backend: Backend selector forwarded to every submodule.
        ending_bias_pair_dims: Source pair axes mapped to ending attention's bias query/key axes.
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
        *,
        ending_bias_pair_dims: tuple[int, int] = (-3, -2),
    ) -> None:
        super().__init__(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
            ending_bias_pair_dims=ending_bias_pair_dims,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=(-2, -1))
        self.columnwise_dropout = StructuredDropout(dropout, broadcast_dims=(-3, -1))

    def triangle_update(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Apply the four triangle residual branches, dropping whole rows and then whole columns.

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
