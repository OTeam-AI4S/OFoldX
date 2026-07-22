r"""Operator backend selection.

``operator_backend`` chooses which implementation an operator uses:

- ``"oops"``      — optional OOps implementation.
- ``"sdpa"``      — ``torch.nn.functional.scaled_dot_product_attention`` (attention only).
- ``"eager"``     — PyTorch implementation.
- ``"auto"``      — OOps when available and suitable, otherwise the public default.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from ._oops import OOPS_AVAILABLE


def oops_available() -> bool:
    r"""
    Return whether OOps imported successfully.

    Runtime ``"auto"`` dispatch also considers the participating tensors.

    Examples:
        >>> from ofoldx.modules.backend.resolve import oops_available
        >>> isinstance(oops_available(), bool)
        True
    """

    return OOPS_AVAILABLE


def resolve_operator_backend(requested: str, *, supported: tuple[str, ...], public_default: str) -> str:
    r"""
    Resolve a backend request without tensor-specific constraints.

    ``"auto"`` becomes ``"oops"`` when OOps is available and supported, otherwise ``public_default``.
    Explicit values are validated and returned unchanged.

    Args:
        requested: The requested backend, ``"auto"`` or one of the known backend names.
        supported: The backends this particular operator implements, in preference order.
        public_default: The public backend used by ``"auto"`` when OOps is unavailable.

    Returns:
        A concrete backend name drawn from ``supported`` (or ``public_default``).

    Raises:
        ValueError: If the requested backend is not implemented by this operator.

    Examples:
        >>> from ofoldx.modules.backend.resolve import resolve_operator_backend
        >>> resolve_operator_backend("eager", supported=("oops", "eager"), public_default="eager")
        'eager'
        >>> resolve_operator_backend("nope", supported=("oops", "eager"), public_default="eager")
        Traceback (most recent call last):
            ...
        ValueError: unsupported operator_backend 'nope'; expected one of: auto, eager, oops
    """

    if requested == "auto":
        return "oops" if "oops" in supported and oops_available() else public_default
    if requested not in supported:
        options = ", ".join(sorted({"auto", *supported}))
        raise ValueError(f"unsupported operator_backend {requested!r}; expected one of: {options}")
    return requested


def resolve_config_operator_backend(requested: str, *, supported: tuple[str, ...], public_default: str) -> str:
    r"""
    Validate a stored backend setting while preserving ``"auto"``.

    Runtime dispatchers resolve ``"auto"`` after tensors are available.

    Args:
        requested: The requested backend, ``"auto"`` or one of the known backend names.
        supported: The backends this particular operator implements, in preference order.
        public_default: The public backend used by runtime ``"auto"`` dispatch.

    Returns:
        ``"auto"`` unchanged, or a concrete validated backend request.

    Examples:
        >>> from ofoldx.modules.backend.resolve import resolve_config_operator_backend
        >>> resolve_config_operator_backend("auto", supported=("oops", "eager"), public_default="eager")
        'auto'
    """

    if requested == "auto":
        return "auto"
    return resolve_operator_backend(requested, supported=supported, public_default=public_default)


def resolve_tensor_operator_backend(
    requested: str,
    *,
    tensors: Iterable[Tensor | None],
    supported: tuple[str, ...],
    public_default: str,
) -> str:
    r"""
    Resolve a backend for concrete tensors.

    ``"auto"`` selects OOps only when it is available, supported, and every provided
    tensor is on CUDA. ``None`` values are ignored and at least one tensor is required.
    Explicit requests are validated without device-based fallback.

    Args:
        requested: The requested backend, ``"auto"`` or one of the known backend names.
        tensors: Tensors considered by ``"auto"`` backend selection. ``None`` values are ignored so
            optional masks can be passed directly.
        supported: The backends this particular operator implements, in preference order.
        public_default: The public backend to use when OOps is unavailable or unsuitable for the tensors.

    Returns:
        A concrete backend name drawn from ``supported`` (or ``public_default``).

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.resolve import resolve_tensor_operator_backend
        >>> resolve_tensor_operator_backend(
        ...     "auto", tensors=(torch.ones(1),), supported=("oops", "eager"), public_default="eager"
        ... )
        'eager'
    """

    if requested == "auto":
        if "oops" in supported and oops_available():
            has_tensor = False
            for tensor in tensors:
                if tensor is None:
                    continue
                has_tensor = True
                if not tensor.is_cuda:
                    break
            else:
                if has_tensor:
                    return "oops"
        return public_default
    return resolve_operator_backend(requested, supported=supported, public_default=public_default)
