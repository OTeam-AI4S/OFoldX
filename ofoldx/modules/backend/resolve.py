r"""Operator backend selection.

Public backend requests have one shared meaning:

- ``"auto"``  — OOps when available and suitable, otherwise PyTorch.
- ``"torch"`` — the operator's canonical PyTorch implementation.
- ``"sdpa"``, ``"oops"``, ``"cueq"``, ``"apex"``, and ``"deepspeed"`` — use that
  provider where it implements the operator and the canonical PyTorch implementation elsewhere.

Low-level numerical diagnostics may also request a concrete implementation such as
``"eager"``. Concrete names are implementation details, not a second public
configuration axis.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from ._oops import OOPS_AVAILABLE

PUBLIC_OPERATOR_BACKENDS = frozenset({"auto", "torch", "sdpa", "oops", "cueq", "apex", "deepspeed"})


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
    ``"torch"`` becomes ``public_default``. Concrete implementation names are validated and returned unchanged.

    Args:
        requested: The requested public backend or a concrete implementation name.
        supported: The backends this particular operator implements, in preference order.
        public_default: The operator's canonical PyTorch implementation.

    Returns:
        A concrete backend name drawn from ``supported`` (or ``public_default``).

    Raises:
        ValueError: If the request is neither a public provider nor a supported concrete implementation.

    Examples:
        >>> from ofoldx.modules.backend.resolve import resolve_operator_backend
        >>> resolve_operator_backend("eager", supported=("oops", "eager"), public_default="eager")
        'eager'
        >>> resolve_operator_backend("nope", supported=("oops", "eager"), public_default="eager")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: unsupported operator_backend 'nope'; expected one of: ...torch
    """

    if requested == "auto":
        return "oops" if "oops" in supported and oops_available() else public_default
    if requested == "torch":
        return public_default
    if requested in PUBLIC_OPERATOR_BACKENDS and requested not in supported:
        return public_default
    if requested not in supported:
        options = ", ".join(sorted({*PUBLIC_OPERATOR_BACKENDS, *supported}))
        raise ValueError(f"unsupported operator_backend {requested!r}; expected one of: {options}")
    return requested


def resolve_config_operator_backend(requested: str, *, supported: tuple[str, ...], public_default: str) -> str:
    r"""
    Validate a stored backend setting while preserving public backend requests.

    Runtime dispatchers resolve every public provider for the concrete operator when it executes.

    Args:
        requested: The requested public backend or a concrete implementation name.
        supported: The backends this particular operator implements, in preference order.
        public_default: The operator's canonical PyTorch implementation.

    Returns:
        A preserved public request or a validated concrete implementation name.

    Examples:
        >>> from ofoldx.modules.backend.resolve import resolve_config_operator_backend
        >>> resolve_config_operator_backend("auto", supported=("oops", "eager"), public_default="eager")
        'auto'
    """

    if requested in PUBLIC_OPERATOR_BACKENDS:
        return requested
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
    tensor is on CUDA. It otherwise selects the canonical PyTorch implementation.
    ``"torch"`` always selects that PyTorch implementation. ``None`` values are ignored
    and at least one tensor is required for automatic OOps selection.

    Args:
        requested: The requested public backend or a concrete implementation name.
        tensors: Tensors considered by ``"auto"`` backend selection. ``None`` values are ignored so
            optional masks can be passed directly.
        supported: The backends this particular operator implements, in preference order.
        public_default: The operator's canonical PyTorch implementation.

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
