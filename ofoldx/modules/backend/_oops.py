r"""
Optional-OOps access seam.

OOps (``ooperators``) provides the fused CUDA kernels behind the fast operator path, but it is
**not** a hard dependency: OFoldX must import and run on the public eager path when OOps is absent
(the private package is not yet released). This module is the *single* guarded import site — every
other module reaches OOps only through it: the fused [`LayerNorm`][torch.nn.LayerNorm] (eager
``torch.nn.LayerNorm`` when OOps is missing), the per-operator kernels via
[`oops_kernel`][ofoldx.modules.backend._oops.oops_kernel], and
[`OOpsUnsupportedError`][ofoldx.modules.backend._oops.OOpsUnsupportedError]. Because the import is
guarded here and nowhere else, ``import ofoldx`` never fails on a missing private package.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

try:
    import oops as _oops
    from oops import LayerNorm, OOpsUnsupportedError

    OOPS_AVAILABLE = True
except ImportError:
    _oops = None
    OOPS_AVAILABLE = False

    # oops.LayerNorm is an API-compatible fused subclass of nn.LayerNorm, so nn.LayerNorm is the exact
    # eager reference (identical constructor and math).
    LayerNorm = nn.LayerNorm

    class OOpsUnsupportedError(RuntimeError):  # type: ignore[no-redef]
        r"""Raised when an OOps-only backend or kernel is requested but the ``oops`` package is unavailable."""


def oops_kernel(name: str) -> Callable[..., object]:
    r"""
    Return a named OOps kernel, or a stub that raises when OOps is unavailable.

    Backend dispatchers bind their fused kernel once at import via this helper. When OOps is installed
    the real kernel is returned; when it is absent a stub is returned that raises
    [`OOpsUnsupportedError`][ofoldx.modules.backend._oops.OOpsUnsupportedError] only if it is actually
    called, so ``import ofoldx`` succeeds and an explicit ``operator_backend="oops"`` request fails with a
    clear message instead of an ``ImportError`` at module load. ``"auto"`` resolution never reaches the
    stub because [`oops_available`][ofoldx.modules.backend.resolve.oops_available] is ``False`` without OOps.

    Args:
        name: Attribute name of the kernel on the ``oops`` package (e.g. ``"transition"``).

    Returns:
        The OOps kernel callable, or a raising stub when OOps is unavailable.

    Examples:
        >>> from ofoldx.modules.backend._oops import oops_kernel
        >>> callable(oops_kernel("transition"))
        True
    """

    if OOPS_AVAILABLE:
        return getattr(_oops, name)

    def _unavailable(*_args: object, **_kwargs: object) -> object:
        raise OOpsUnsupportedError(
            f"operator_backend='oops' requires the optional 'oops' package (kernel {name!r} is unavailable)"
        )

    return _unavailable
