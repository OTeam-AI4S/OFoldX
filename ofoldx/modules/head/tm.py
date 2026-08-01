r"""
Predicted TM-score mathematics.
"""

from torch import Tensor


def tm_length_scale(num_tokens: Tensor | float) -> Tensor | float:
    r"""Return ``d0(n) = 1.24 * (max(n, 19) - 15) ** (1 / 3) - 1.8``."""

    if isinstance(num_tokens, Tensor):
        num_tokens = num_tokens.clamp(min=19)
    else:
        num_tokens = max(float(num_tokens), 19.0)
    return 1.24 * (num_tokens - 15.0) ** (1.0 / 3.0) - 1.8


def tm_bin_weights(bin_centers: Tensor, num_tokens: Tensor | float) -> Tensor:
    r"""Return the per-bin weight ``1 / (1 + (center / d0(n)) ** 2)``."""

    scale = tm_length_scale(num_tokens)
    if isinstance(scale, Tensor):
        scale = scale.unsqueeze(-1)
    return 1.0 / (1.0 + (bin_centers / scale) ** 2)


def tm_weight_from_pae(pae_logits: Tensor, num_tokens: Tensor | float, *, bin_centers: Tensor) -> Tensor:
    r"""Return the expected per-pair TM weight from PAE logits."""

    probability = pae_logits.softmax(dim=-1)
    return (probability * tm_bin_weights(bin_centers, num_tokens)).sum(dim=-1)
