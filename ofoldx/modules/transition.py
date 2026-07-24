r"""
Transition update branches and row-wise dropout for trunk and atom blocks.
"""

from __future__ import annotations

from torch import Tensor, nn

from .adaptive_norm import AdaptiveLayerNorm
from .backend import LayerNorm, resolve_config_operator_backend
from .backend import transition as dispatch_transition


class Transition(nn.Module):
    r"""
    Pre-normalized position-wise SwiGLU transition.

    Layer-norms the input, then applies a gated linear unit: the SiLU-activated ``gate_projection``
    modulates the ``value_projection`` before ``output_projection`` maps back to ``hidden_size``.
    The computation is dispatched through
    [ofoldx.modules.backend][], which selects the OOps-owned or eager implementation.
    It returns an update; the containing block owns the residual addition.

    Args:
        hidden_size: Input and output hidden width.
        expansion: Hidden expansion factor; the inner width is ``expansion * hidden_size``.
        operator_backend: Which operator backend to run; see [ofoldx.modules.backend][].
    """

    def __init__(self, hidden_size: int, expansion: int = 4, operator_backend: str = "auto") -> None:
        super().__init__()
        self.operator_backend = resolve_config_operator_backend(
            operator_backend, supported=("oops", "eager"), public_default="eager"
        )
        expanded_size = expansion * hidden_size
        self.input_norm = LayerNorm(hidden_size, operator_backend=operator_backend)
        self.gate_projection = nn.Linear(hidden_size, expanded_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, expanded_size, bias=False)
        self.output_projection = nn.Linear(expanded_size, hidden_size, bias=False)

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
            norm_weight=self.input_norm.weight,
            norm_bias=self.input_norm.bias,
            gate_weight=self.gate_projection.weight,
            up_weight=self.value_projection.weight,
            down_weight=self.output_projection.weight,
            eps=self.input_norm.eps,
            operator_backend=self.operator_backend,
        )


class RowwiseDropout(nn.Module):
    r"""
    Apply dropout with a mask shared across the row axis.

    For an input shaped ``(..., num_rows, num_columns, hidden_size)``, the sampled
    mask has shape ``(..., 1, num_columns, hidden_size)`` and is broadcast to every
    row. Evaluation mode and zero probability are identities.

    Args:
        probability: Dropout probability for each element of the broadcast mask.
    """

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(probability)

    def forward(self, hidden: Tensor) -> Tensor:
        r"""
        Apply a dropout mask broadcast across the ``-3`` axis.

        Args:
            hidden: Tensor whose row axis is ``-3``.

        Returns:
            A tensor with the same shape as ``hidden``.
        """

        # Dropout is an identity in eval (and at p=0); short-circuit so we neither waste work nor
        # build a padded mask that would not align with a NestedTensor's ragged structure.
        if not self.training or self.dropout.p == 0:
            return hidden
        shape = list(hidden.shape)
        shape[-3] = 1
        mask = hidden.new_ones(shape)
        return hidden * self.dropout(mask)


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
        operator_backend: Public backend request forwarded to the adaptive norm.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        conditioning_size: int = 384,
        expansion: int = 2,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        expanded_size = expansion * hidden_size
        self.conditioned_norm = AdaptiveLayerNorm(
            hidden_size,
            conditioning_size,
            operator_backend=operator_backend,
        )
        self.gate_projection = nn.Linear(hidden_size, expanded_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, expanded_size, bias=False)
        self.output_projection = nn.Linear(expanded_size, hidden_size, bias=False)
        self.conditioning_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.silu = nn.SiLU()

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
        output = self.output_projection(
            self.silu(self.gate_projection(hidden)) * self.value_projection(hidden),
        )
        return output * self.sigmoid(self.conditioning_gate_projection(conditioning))
