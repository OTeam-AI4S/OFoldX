r"""
Backend selection and validation.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from ._oops import OOPS_AVAILABLE

PUBLIC_BACKENDS = frozenset({"auto", "eager", "sdpa", "oops", "cueq", "apex", "deepspeed"})


def oops_available() -> bool:
    r"""
    Return whether OOps imported successfully.

    Runtime ``"auto"`` dispatch also considers the participating tensors.

    Returns:
        Whether the optional OOps package is available.
    """

    return OOPS_AVAILABLE


def resolve_backend(requested: str, *, supported: tuple[str, ...], public_default: str) -> str:
    r"""
    Resolve a backend request without tensor-specific constraints.

    ``"auto"`` becomes ``"oops"`` when OOps is available and supported, otherwise
    ``public_default``. Explicit implementation names are validated and returned
    unchanged.

    Args:
        requested: The requested backend.
        supported: The backends this particular operator implements, in preference order.
        public_default: The operator's default PyTorch implementation.

    Returns:
        A concrete backend name drawn from ``supported`` (or ``public_default``).

    Raises:
        ValueError: If the requested backend is not supported.
    """

    if requested == "auto":
        return "oops" if "oops" in supported and oops_available() else public_default
    if requested in PUBLIC_BACKENDS and requested not in supported:
        return public_default
    if requested not in supported:
        options = ", ".join(sorted({*PUBLIC_BACKENDS, *supported}))
        raise ValueError(f"unsupported backend {requested!r}; expected one of: {options}")
    return requested


def resolve_config_backend(requested: str, *, supported: tuple[str, ...], public_default: str) -> str:
    r"""
    Validate a stored backend setting while preserving public backend requests.

    Runtime dispatchers resolve every public provider for the concrete operator when it executes.

    Args:
        requested: The requested backend.
        supported: The backends this particular operator implements, in preference order.
        public_default: The operator's default PyTorch implementation.

    Returns:
        A preserved automatic request or a validated concrete implementation name.
    """

    if requested in PUBLIC_BACKENDS:
        return requested
    return resolve_backend(requested, supported=supported, public_default=public_default)


def resolve_tensor_backend(
    requested: str,
    *,
    tensors: Iterable[Tensor | None],
    supported: tuple[str, ...],
    public_default: str,
) -> str:
    r"""
    Resolve a backend for concrete tensors.

    ``"auto"`` selects OOps only when it is available, supported, and every provided
    tensor is on CUDA. It otherwise selects the eager implementation. ``None``
    values are ignored and at least one tensor is required for automatic OOps
    selection.

    Args:
        requested: The requested backend.
        tensors: Tensors considered by ``"auto"`` backend selection. ``None`` values are ignored so
            optional masks can be passed directly.
        supported: The backends this particular operator implements, in preference order.
        public_default: The operator's default PyTorch implementation.

    Returns:
        A concrete backend name drawn from ``supported`` (or ``public_default``).
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
    return resolve_backend(requested, supported=supported, public_default=public_default)
