r"""
Backend dispatch for class-last cross entropy used by the loss heads.

The dispatcher owns everything backend-specific for this operator in one place: the supported
backends, ``"auto"`` resolution, and each backend's calling convention. Modules pass normalized
inputs plus their requested ``operator_backend`` and stay backend-agnostic. New backends
(cuEquivariance / NVIDIA) are added as a branch here, not in every caller.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from oops import OOpsUnsupportedError
from oops import cross_entropy as _oops_cross_entropy
from torch import Tensor

from .resolve import resolve_tensor_operator_backend

_SUPPORTED = ("oops", "eager")
_INTEGER_LABEL_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def cross_entropy(logits: Tensor, labels: Tensor, *, operator_backend: str) -> Tensor:
    r"""
    Return unreduced class-last cross entropy over the final axis on the selected backend.

    ``labels`` may be a probability/one-hot tensor with the same shape as ``logits`` or an integer
    class-index tensor with shape ``logits.shape[:-1]``; the integer path is mathematically equivalent
    to one-hot labels but avoids materializing the class axis. Output shape is ``logits.shape[:-1]``.

    Args:
        logits: ``[..., num_classes]`` class logits with the class axis last.
        labels: Probability labels shaped like ``logits`` or integer labels shaped ``logits.shape[:-1]``.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.

    Returns:
        ``logits.shape[:-1]`` unreduced per-example cross entropy.

    Raises:
        OOpsUnsupportedError: If the resolved backend is ``"oops"`` on non-CUDA tensors.
        ValueError: If ``labels`` is neither shaped like ``logits`` nor ``logits.shape[:-1]``.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.cross_entropy import cross_entropy
        >>> logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3]])
        >>> cross_entropy(logits, torch.tensor([0, 1]), operator_backend="eager").round(decimals=4)
        tensor([0.3168, 0.4534])
        >>> one_hot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> cross_entropy(logits, one_hot, operator_backend="eager").round(decimals=4)
        tensor([0.3168, 0.4534])
    """

    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(logits, labels),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        if not logits.is_cuda:
            raise OOpsUnsupportedError("operator_backend 'oops' for cross_entropy requires CUDA tensors")
        return _oops_cross_entropy(logits, labels)
    # Eager reference, numerically equivalent to oops.cross_entropy: the probability/one-hot path and
    # the integer-index path both reduce to the standard log-softmax cross entropy along the last axis.
    if labels.shape == logits.shape:
        log_probs = F.log_softmax(logits, dim=-1)
        return -(labels.to(dtype=log_probs.dtype) * log_probs).sum(dim=-1)
    if labels.shape == logits.shape[:-1] and labels.dtype in _INTEGER_LABEL_DTYPES:
        log_probs = F.log_softmax(logits, dim=-1)
        return -log_probs.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    raise ValueError("labels must have shape logits.shape or logits.shape[:-1]")
