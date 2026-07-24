r"""
Guarded access to optional OOps operators.

Named kernels raise only when called, allowing public PyTorch paths to remain available.
"""

from __future__ import annotations

from typing import Callable

OOpsUnsupportedError: type[NotImplementedError]

try:
    import oops as _oops
    from oops import OOpsUnsupportedError as _ImportedOOpsUnsupportedError

    OOpsUnsupportedError = _ImportedOOpsUnsupportedError
    OOPS_AVAILABLE = True
except ModuleNotFoundError as error:
    if error.name != "oops":
        raise
    _oops = None
    OOPS_AVAILABLE = False

    class _MissingOOpsUnsupportedError(NotImplementedError):
        r"""
        Raised when an OOps kernel is requested but OOps is unavailable.
        """

    OOpsUnsupportedError = _MissingOOpsUnsupportedError


def oops_kernel(name: str) -> Callable[..., object]:
    r"""
    Return a named OOps kernel or a deferred unavailable-kernel error.

    The unavailable stub raises [`OOpsUnsupportedError`][] when called. ``"auto"``
    dispatch excludes OOps when the package is unavailable.

    Args:
        name: Attribute name of the public callable on the ``oops`` package (e.g. ``"transition"``).

    Returns:
        The OOps-owned implementation callable, or a raising stub when OOps is unavailable.
    """

    if OOPS_AVAILABLE:
        return getattr(_oops, name)

    def _unavailable(*_args: object, **_kwargs: object) -> object:
        raise OOpsUnsupportedError(
            f"operator_backend='oops' requires the optional 'oops' package (kernel {name!r} is unavailable)"
        )

    return _unavailable
