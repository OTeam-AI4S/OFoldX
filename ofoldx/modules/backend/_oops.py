r"""Guarded access to optional OOps operators.

Without OOps, :class:`LayerNorm` uses PyTorch and named kernels raise only when
called, allowing public eager paths to remain available.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

OOpsUnsupportedError: type[NotImplementedError]

try:
    import oops as _oops
    from oops import LayerNorm
    from oops import OOpsUnsupportedError as _ImportedOOpsUnsupportedError

    OOpsUnsupportedError = _ImportedOOpsUnsupportedError
    OOPS_AVAILABLE = True
except ModuleNotFoundError as error:
    if error.name != "oops":
        raise
    _oops = None
    OOPS_AVAILABLE = False

    class _MissingOOpsUnsupportedError(NotImplementedError):
        r"""Raised when an OOps kernel is requested but OOps is unavailable."""

    OOpsUnsupportedError = _MissingOOpsUnsupportedError
    LayerNorm = nn.LayerNorm


def oops_kernel(name: str) -> Callable[..., object]:
    r"""
    Return a named OOps kernel or a deferred unavailable-kernel error.

    The unavailable stub raises :class:`OOpsUnsupportedError` when called. ``"auto"``
    dispatch excludes OOps when the package is unavailable.

    Args:
        name: Attribute name of the public callable on the ``oops`` package (e.g. ``"transition"``).

    Returns:
        The OOps-owned implementation callable, or a raising stub when OOps is unavailable.

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
