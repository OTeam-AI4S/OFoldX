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
Pre-normalized and conditioned transition updates.
"""

from __future__ import annotations

from torch import Tensor, nn

from .adaptive_norm import AdaptiveLayerNorm
from .backend import LayerNorm, resolve_config_backend
from .backend import transition as dispatch_transition
from .feed_forward import FeedForward, SwiGluFeedForward


class FeedForwardTransition(nn.Module):
    r"""
    Pre-normalized two-projection feed-forward transition returning an update.

    Args:
        hidden_size: Width of input and output representations.
        expansion: Intermediate-width multiplier.
        backend: Backend used by input normalization.
        dropout: Dropout probability within the feed-forward.
        activation: Activation used by the feed-forward; defaults to ``"relu"``.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        expansion: int,
        backend: str = "auto",
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_norm = LayerNorm(hidden_size, backend=backend)
        self.feed_forward = FeedForward(hidden_size, hidden_size * expansion, dropout=dropout, activation=activation)

    def forward(self, hidden: Tensor, mask: Tensor | None = None) -> Tensor:
        r"""
        Return a masked additive transition update.

        Args:
            hidden: Hidden representation with ``hidden_size`` channels.
            mask: Optional validity mask matching ``hidden`` without its channel axis.

        Returns:
            Transition update with the same shape as ``hidden``.
        """

        output = self.feed_forward(self.input_norm(hidden))
        if mask is None:
            return output
        return output * mask.unsqueeze(-1).to(dtype=output.dtype)


class Transition(nn.Module):
    r"""
    Pre-normalized position-wise SwiGLU transition.

    Layer-norms the input, then applies a
    [SwiGluFeedForward][ofoldx.modules.feed_forward.SwiGluFeedForward].
    The computation is dispatched through
    [ofoldx.modules.backend][], which selects the OOps-owned or eager implementation.
    It returns an update; the containing block owns the residual addition.

    Args:
        hidden_size: Input and output hidden width.
        expansion: Hidden expansion factor; the inner width is ``expansion * hidden_size``.
        backend: Which backend to run; see [ofoldx.modules.backend][].
    """

    def __init__(self, hidden_size: int, expansion: int = 4, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        expanded_size = expansion * hidden_size
        self.input_norm = LayerNorm(hidden_size, backend=backend)
        self.feed_forward = SwiGluFeedForward(hidden_size, expanded_size)

    def forward(self, hidden: Tensor) -> Tensor:
        r"""
        Normalize, gate, and project one transition update.

        Args:
            hidden: `[..., hidden_size]` input activations.

        Returns:
            Transition output with the same shape as ``hidden``.
        """

        return dispatch_transition(
            hidden,
            input_norm_weight=self.input_norm.weight,
            input_norm_bias=self.input_norm.bias,
            gate_weight=self.feed_forward.gate_projection.weight,
            up_weight=self.feed_forward.value_projection.weight,
            down_weight=self.feed_forward.output_projection.weight,
            input_norm_eps=self.input_norm.eps,
            backend=self.backend,
        )


class ConditionedTransition(nn.Module):
    r"""
    Position-wise SwiGLU transition modulated by a conditioning tensor.

    Like [Transition][ofoldx.modules.transition.Transition], but the input norm is an adaptive
    (conditioned) layer norm and the SwiGLU output is scaled by a sigmoid gate projected from the
    conditioning tensor, letting the conditioning both shift the normalization and gate the update.
    It returns an update; the containing block owns the residual addition.

    Args:
        hidden_size: Input and output hidden width.
        conditioning_size: Conditioning tensor width.
        expansion: Hidden expansion factor; the inner width is ``expansion * hidden_size``.
        backend: Public backend request forwarded to the adaptive norm.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        conditioning_size: int = 384,
        expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        expanded_size = expansion * hidden_size
        self.conditioned_norm = AdaptiveLayerNorm(
            hidden_size,
            conditioning_size,
            backend=backend,
        )
        self.feed_forward = SwiGluFeedForward(hidden_size, expanded_size)
        self.conditioning_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        r"""
        Adaptive-norm, gate, project, then scale the update by a conditioning-derived sigmoid gate.

        Args:
            hidden: `[..., hidden_size]` input activations.
            conditioning: `[..., conditioning_size]` conditioning activations that drive both the
                adaptive norm and the output gate.

        Returns:
            Conditioned transition output with the same shape as ``hidden``.
        """

        hidden = self.conditioned_norm(hidden, conditioning)
        output = self.feed_forward(hidden)
        return output * self.sigmoid(self.conditioning_gate_projection(conditioning))


class TripleProductConditionedTransition(nn.Module):
    r"""
    Conditioning-gated SwiGLU transition multiplied by a third expansion branch.
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        inner_size = hidden_size * expansion
        self.conditioned_norm = AdaptiveLayerNorm(hidden_size, conditioning_size, backend=backend)
        self.gate_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.silu = nn.SiLU()
        self.value_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.expand_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.conditioning_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.output_projection = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        normalized = self.conditioned_norm(hidden, conditioning)
        activated = (
            self.silu(self.gate_projection(normalized))
            * self.value_projection(normalized)
            * self.expand_projection(normalized)
        )
        gate = self.sigmoid(self.conditioning_gate_projection(conditioning))
        return gate * self.output_projection(activated)
