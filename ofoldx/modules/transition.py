r"""
Transition blocks and the companion row-wise dropout used in trunk/atom blocks.
"""

from __future__ import annotations

from oops import LayerNorm
from torch import Tensor, nn

from .adaptive_norm import BoltzAdaptiveLayerNorm
from .backend import resolve_config_operator_backend
from .backend import transition as dispatch_transition


class Transition(nn.Module):
    r"""
    Position-wise SwiGLU transition block.

    Layer-norms the input, then applies a gated linear unit: the SiLU-activated ``gate_projection``
    modulates the ``value_projection`` before ``output_projection`` maps back to ``hidden_size``.
    The computation is dispatched through
    [ofoldx.modules.backend][], so it can run on the fused ``oops`` kernel or an eager fallback.

    Args:
        hidden_size: Input and output hidden width.
        expansion: Hidden expansion factor; the inner width is ``expansion * hidden_size``.
        operator_backend: Which operator backend to run; see [ofoldx.modules.backend][].

    Examples:
        >>> import torch
        >>> transition = Transition(8, expansion=2, operator_backend="eager")
        >>> transition(torch.randn(2, 3, 8)).shape
        torch.Size([2, 3, 8])
    """

    def __init__(self, hidden_size: int, expansion: int = 4, operator_backend: str = "auto") -> None:
        super().__init__()
        self.operator_backend = resolve_config_operator_backend(
            operator_backend, supported=("oops", "eager"), public_default="eager"
        )
        expanded_size = expansion * hidden_size
        self.norm_input = LayerNorm(hidden_size)
        self.gate_projection = nn.Linear(hidden_size, expanded_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, expanded_size, bias=False)
        self.output_projection = nn.Linear(expanded_size, hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        r"""
        Normalize, gate, and project the input through the transition block.

        Args:
            hidden: `[..., hidden_size]` input activations.

        Returns:
            Transition output with the same shape as ``hidden``.

        Raises:
            OOpsUnsupportedError: If the resolved backend is ``"oops"`` on non-CUDA tensors.

        Examples:
            >>> import torch
            >>> transition = Transition(8, expansion=2, operator_backend="eager")
            >>> transition(torch.randn(1, 4, 8)).shape
            torch.Size([1, 4, 8])
        """

        return dispatch_transition(
            hidden,
            norm_weight=self.norm_input.weight,
            norm_bias=self.norm_input.bias,
            gate_weight=self.gate_projection.weight,
            up_weight=self.value_projection.weight,
            down_weight=self.output_projection.weight,
            eps=self.norm_input.eps,
            operator_backend=self.operator_backend,
        )


class RowwiseDropout(nn.Module):
    r"""
    Dropout that shares one mask across the row dimension of pair/MSA updates.

    Standard dropout would zero elements independently; this variant draws a single mask along the
    ``-3`` (row) axis and broadcasts it, so an entire row is dropped together. It is a no-op in
    eval mode and when ``probability`` is ``0``.

    Args:
        probability: Dropout probability applied per row.

    Examples:
        >>> import torch
        >>> dropout = RowwiseDropout(0.5).eval()
        >>> hidden = torch.randn(2, 3, 4, 5)
        >>> torch.equal(dropout(hidden), hidden)
        True
    """

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(probability)

    def forward(self, hidden: Tensor) -> Tensor:
        r"""
        Drop whole rows of the update, sharing one mask across the ``-3`` axis.

        Args:
            hidden: Pair/MSA update tensor with the row axis at ``-3``.

        Returns:
            Tensor with a dropout mask shared across the row dimension during training, or
            ``hidden`` unchanged in eval mode or when ``probability`` is ``0``.

        Examples:
            >>> import torch
            >>> dropout = RowwiseDropout(0.5).eval()
            >>> hidden = torch.randn(2, 3, 4, 5)
            >>> torch.equal(dropout(hidden), hidden)
            True
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

    Args:
        hidden_size: Input and output hidden width.
        conditioning_size: Conditioning tensor width.
        expansion: Hidden expansion factor; the inner width is ``expansion * hidden_size``.

    Examples:
        >>> import torch
        >>> transition = ConditionedTransition(8, 4, expansion=2)
        >>> hidden = torch.randn(2, 3, 8)
        >>> conditioning = torch.randn(2, 3, 4)
        >>> transition(hidden, conditioning).shape
        torch.Size([2, 3, 8])
    """

    def __init__(
        self,
        hidden_size: int = 768,
        conditioning_size: int = 384,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        expanded_size = expansion * hidden_size
        self.conditioned_norm = BoltzAdaptiveLayerNorm(hidden_size, conditioning_size)
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

        Examples:
            >>> import torch
            >>> transition = ConditionedTransition(8, 4, expansion=2)
            >>> hidden = torch.randn(1, 5, 8)
            >>> conditioning = torch.randn(1, 5, 4)
            >>> transition(hidden, conditioning).shape
            torch.Size([1, 5, 8])
        """

        hidden = self.conditioned_norm(hidden, conditioning)
        output = self.output_projection(
            self.silu(self.gate_projection(hidden)) * self.value_projection(hidden),
        )
        return output * self.sigmoid(self.conditioning_gate_projection(conditioning))
