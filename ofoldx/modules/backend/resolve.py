r"""
Operator-backend resolution shared by the per-operator dispatchers.

``operator_backend`` chooses which implementation an operator uses:

- ``"oops"``      — the OOps fused CUDA kernels (fastest; requires CUDA).
- ``"sdpa"``      — ``torch.nn.functional.scaled_dot_product_attention`` (attention only).
- ``"eager"``     — pure-PyTorch reference (always available).
- ``"auto"``      — resolve at runtime to ``"oops"`` when available, else the best public backend.

``"auto"`` is the default so an install without the (currently private) OOps package still runs on
the public path, while an internal install with OOps + CUDA transparently gets the fast kernels.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from ._oops import OOPS_AVAILABLE


def oops_available() -> bool:
    r"""
    Report whether the optional OOps adapter imported successfully.

    Device eligibility belongs to each actual tensor call. This predicate reports only adapter
    availability; runtime dispatch separately requires the participating tensors to be on CUDA.

    Returns:
        ``True`` when the OOps adapter imported successfully, else ``False``.

    Examples:
        >>> from ofoldx.modules.backend.resolve import oops_available
        >>> isinstance(oops_available(), bool)
        True
    """

    return OOPS_AVAILABLE


def resolve_operator_backend(requested: str, *, supported: tuple[str, ...], public_default: str) -> str:
    r"""
    Resolve an ``operator_backend`` request to a concrete backend this operator supports.

    ``"auto"`` becomes ``"oops"`` when OOps is available and supported, otherwise ``public_default``.
    An explicit value is validated against ``supported`` and returned unchanged. Explicit requests never
    silently degrade to a different implementation.

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
    Validate an operator-backend setting while preserving ``"auto"`` for runtime dispatch.

    Module constructors use this when they need to store the user's backend preference. Resolving
    ``"auto"`` at construction time can be wrong because the module might later be called with CPU tensors
    even on a CUDA-visible host. Runtime dispatchers should use [`resolve_tensor_operator_backend`][].

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
    Resolve an operator backend for a specific call site with concrete tensors.

    This is the runtime companion to [`resolve_operator_backend`][]. ``"auto"`` only selects the OOps
    backend when OOps is available *and* every provided tensor is already on CUDA. Explicit requests keep
    the stricter behavior: ``operator_backend="oops"`` still resolves to ``"oops"`` even for CPU tensors,
    so the dispatcher can raise the usual device error.

    Args:
        requested: The requested backend, ``"auto"`` or one of the known backend names.
        tensors: Tensors that must be CUDA-resident for the OOps implementation to run. ``None`` values
            are ignored so optional masks can be passed directly.
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
