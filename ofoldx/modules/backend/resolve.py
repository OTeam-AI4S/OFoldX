r"""
Operator-backend resolution shared by the per-operator dispatchers.

``operator_backend`` chooses which implementation an operator uses:

- ``"oops"``      — the OOps fused CUDA kernels (fastest; requires CUDA).
- ``"sdpa"``      — ``torch.nn.functional.scaled_dot_product_attention`` (attention only).
- ``"deepspeed"`` — DS4Sci ``EvoformerAttention`` (attention only).
- ``"eager"``     — pure-PyTorch reference (always available).
- ``"auto"``      — resolve at runtime to ``"oops"`` when available, else the best public backend.

``"auto"`` is the default so an install without the (currently private) OOps package still runs on
the public path, while an internal install with OOps + CUDA transparently gets the fast kernels.
cuEquivariance / NVIDIA backends slot in by adding them to a dispatcher's ``supported`` tuple.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable

from ._oops import OOPS_AVAILABLE

# Every backend the library knows about. A request in this set that a given operator does not
# implement (e.g. "sdpa" for a non-attention op) falls back to that operator's public default rather
# than erroring; only values outside this set are treated as typos and rejected.
_KNOWN_BACKENDS = frozenset({"auto", "oops", "sdpa", "deepspeed", "eager"})


@functools.lru_cache(maxsize=1)
def oops_available() -> bool:
    r"""
    Report whether the OOps fused kernels can actually run in this process.

    Both conditions must hold: the (currently private) ``oops`` package imports, and a CUDA device is
    visible. The result is cached for the process lifetime, so toggling CUDA visibility afterwards is
    not observed. This is the predicate ``"auto"`` resolution consults to decide between the fast OOps
    path and the public fallback.

    Returns:
        ``True`` when ``import oops`` succeeds and ``torch.cuda.is_available()``, else ``False``.

    Examples:
        >>> from ofoldx.modules.backend.resolve import oops_available
        >>> isinstance(oops_available(), bool)
        True
    """

    if not OOPS_AVAILABLE:
        return False
    import torch

    return torch.cuda.is_available()


def resolve_operator_backend(requested: str, *, supported: tuple[str, ...], public_default: str) -> str:
    r"""
    Resolve an ``operator_backend`` request to a concrete backend this operator supports.

    ``"auto"`` becomes ``"oops"`` when OOps is available and supported, otherwise ``public_default``.
    An explicit value is validated against ``supported`` and returned unchanged. A request that names a
    known backend the operator does not implement (e.g. ``"sdpa"`` for a non-attention op) degrades to
    ``public_default`` rather than raising; only genuinely unknown names are rejected as typos.

    Args:
        requested: The requested backend, ``"auto"`` or one of the known backend names.
        supported: The backends this particular operator implements, in preference order.
        public_default: The backend to fall back to when OOps is unavailable or ``requested`` names a
            known-but-unimplemented backend.

    Returns:
        A concrete backend name drawn from ``supported`` (or ``public_default``).

    Raises:
        ValueError: If ``requested`` is neither ``"auto"`` nor any known backend name.

    Examples:
        >>> from ofoldx.modules.backend.resolve import resolve_operator_backend
        >>> resolve_operator_backend("eager", supported=("oops", "eager"), public_default="eager")
        'eager'
        >>> resolve_operator_backend("sdpa", supported=("oops", "eager"), public_default="eager")
        'eager'
        >>> resolve_operator_backend("nope", supported=("oops", "eager"), public_default="eager")
        Traceback (most recent call last):
            ...
        ValueError: unknown operator_backend 'nope'; expected one of: auto, deepspeed, eager, oops, sdpa
    """

    if requested == "auto":
        return "oops" if "oops" in supported and oops_available() else public_default
    if requested in supported:
        return requested
    if requested in _KNOWN_BACKENDS:
        # A valid backend this operator does not implement (e.g. "sdpa" for a non-attention op):
        # fall back to this operator's public default instead of erroring.
        return public_default
    options = ", ".join(sorted({*supported, *_KNOWN_BACKENDS}))
    raise ValueError(f"unknown operator_backend {requested!r}; expected one of: {options}")


def resolve_config_operator_backend(requested: str, *, supported: tuple[str, ...], public_default: str) -> str:
    r"""
    Validate an operator-backend setting while preserving ``"auto"`` for runtime dispatch.

    Module constructors use this when they need to store the user's backend preference. Resolving
    ``"auto"`` at construction time can be wrong because the module might later be called with CPU tensors
    even on a CUDA-visible host. Runtime dispatchers should use [`resolve_tensor_operator_backend`][].

    Args:
        requested: The requested backend, ``"auto"`` or one of the known backend names.
        supported: The backends this particular operator implements, in preference order.
        public_default: The public backend to use for known-but-unimplemented backend names.

    Returns:
        ``"auto"`` unchanged, or a concrete validated backend request.

    Examples:
        >>> from ofoldx.modules.backend.resolve import resolve_config_operator_backend
        >>> resolve_config_operator_backend("auto", supported=("oops", "eager"), public_default="eager")
        'auto'
        >>> resolve_config_operator_backend("sdpa", supported=("oops", "eager"), public_default="eager")
        'eager'
    """

    if requested == "auto":
        return "auto"
    return resolve_operator_backend(requested, supported=supported, public_default=public_default)


def resolve_tensor_operator_backend(
    requested: str,
    *,
    tensors: Iterable[object | None],
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
        if "oops" in supported and oops_available() and _all_cuda_tensors(tensors):
            return "oops"
        return public_default
    return resolve_operator_backend(requested, supported=supported, public_default=public_default)


def _all_cuda_tensors(tensors: Iterable[object | None]) -> bool:
    has_tensor = False
    for tensor in tensors:
        if tensor is None:
            continue
        has_tensor = True
        if not getattr(tensor, "is_cuda", False):
            return False
    return has_tensor
