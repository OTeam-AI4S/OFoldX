r"""
Predicted-TM (pTM / ipTM) inner kernel.

This module owns the length-dependent TM-score normalization ``d0(n)`` and the
per-bin weight ``1 / (1 + (d / d0)^2)``. Callers own all outer reductions:
which token pairs enter an average, which token count feeds ``d0``, any interface
masks, and row- or frame-wise maxima.
"""

from __future__ import annotations

from torch import Tensor

# d0(n) = _D0_SCALE * (max(n, _D0_MIN_TOKENS) - _D0_OFFSET) ** (1/3) - _D0_SHIFT
_D0_SCALE = 1.24
_D0_OFFSET = 15.0
_D0_SHIFT = 1.8
_D0_MIN_TOKENS = 19.0


def tm_length_scale(num_tokens: Tensor | float, *, min_tokens: float = _D0_MIN_TOKENS) -> Tensor | float:
    r"""
    Return the TM-score length normalization ``d0(n) = 1.24 * (max(n, 19) - 15)^(1/3) - 1.8``.

    Args:
        num_tokens: Effective token count ``n`` -- a Python scalar or a tensor of counts.
        min_tokens: Lower clamp on ``n`` before the cube root.

    Returns:
        ``d0`` with the same kind as ``num_tokens`` (float in, float out; tensor in, tensor out).

    Examples:
        >>> round(float(tm_length_scale(19)), 5)
        0.16838
        >>> import torch
        >>> tm_length_scale(torch.tensor([19.0, 100.0])).shape
        torch.Size([2])
    """

    if isinstance(num_tokens, Tensor):
        clipped: Tensor | float = num_tokens.clamp(min=float(min_tokens))
    else:
        clipped = float(max(num_tokens, min_tokens))
    return _D0_SCALE * (clipped - _D0_OFFSET) ** (1.0 / 3.0) - _D0_SHIFT


def tm_bin_weights(bin_centers: Tensor, num_tokens: Tensor | float) -> Tensor:
    r"""
    Per-bin TM-score weight ``1 / (1 + (center / d0(n))^2)``.

    When ``num_tokens`` is a tensor its ``d0`` is unsqueezed on a trailing bin axis so the weight
    broadcasts against a ``(..., num_bins)`` distribution; the caller supplies ``num_tokens`` at the
    rank required by its outer reduction, such as a per-pair count, a broadcast batch count, or a scalar.

    Args:
        bin_centers: ``[num_bins]`` distance-error bin centers, in Angstrom.
        num_tokens: Effective token count ``n`` feeding ``d0`` -- scalar or count tensor.

    Returns:
        The per-bin TM-score weight, broadcastable to the softmaxed PAE distribution.

    Examples:
        >>> import torch
        >>> tm_bin_weights(torch.linspace(0.5, 31.5, 32), 128).shape
        torch.Size([32])
    """

    d0 = tm_length_scale(num_tokens)
    if isinstance(d0, Tensor):
        d0 = d0.unsqueeze(-1)
    return 1.0 / (1.0 + (bin_centers / d0) ** 2)


def tm_weight_from_pae(pae_logits: Tensor, num_tokens: Tensor | float, *, bin_centers: Tensor) -> Tensor:
    r"""
    Full inner kernel: softmax ``pae_logits`` over bins, weight by [`tm_bin_weights`][], and sum.

    Uses a plain ``softmax(-1)``. Callers that upcast to float around the softmax or reuse a
    pre-decoded probability tensor call [`tm_bin_weights`][] directly on their own distribution
    instead of this convenience wrapper.

    Args:
        pae_logits: ``[..., num_bins]`` predicted-aligned-error logits.
        num_tokens: Effective token count ``n`` feeding ``d0``.
        bin_centers: ``[num_bins]`` distance-error bin centers, in Angstrom.

    Returns:
        ``[...]`` TM-adjusted PAE (the expected per-token-pair TM-score contribution).

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> tm_weight_from_pae(torch.randn(2, 3, 3, 32), 128, bin_centers=torch.linspace(0.5, 31.5, 32)).shape
        torch.Size([2, 3, 3])
    """

    probability = pae_logits.softmax(dim=-1)
    return (probability * tm_bin_weights(bin_centers, num_tokens)).sum(dim=-1)
