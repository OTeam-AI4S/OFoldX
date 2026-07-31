r"""
Guarded access to optional OOps operators.

Missing-package errors are deferred until a named kernel is called, allowing public
PyTorch paths to remain available.
"""

from __future__ import annotations

from typing import Callable

try:
    import oops as _oops

    OOPS_AVAILABLE = True
except ModuleNotFoundError as error:
    if error.name != "oops":
        raise
    _oops = None
    OOPS_AVAILABLE = False


def oops_kernel(name: str) -> Callable[..., object]:
    r"""
    Return a named OOps kernel or a deferred unavailable-kernel error.

    The unavailable stub raises [`ModuleNotFoundError`][] when called. ``"auto"``
    dispatch excludes OOps when the package is unavailable.

    Args:
        name: Attribute name of the public callable on the ``oops`` package (e.g. ``"transition"``).

    Returns:
        The OOps-owned implementation callable, or a raising stub when OOps is unavailable.
    """

    if OOPS_AVAILABLE:
        return getattr(_oops, name)

    def _unavailable(*_args: object, **_kwargs: object) -> object:
        raise ModuleNotFoundError(
            f"backend='oops' requires the optional 'oops' package (kernel {name!r} is unavailable)",
            name="oops",
        )

    return _unavailable
