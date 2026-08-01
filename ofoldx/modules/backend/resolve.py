# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Backend selection and validation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib.util import find_spec
from typing import Literal, get_args

from torch import Tensor

from ..._oops import MissingOopsKernel
from ...errors import RequestRef, fail

CUEQ_AVAILABLE = find_spec("cuequivariance_torch") is not None
APEX_AVAILABLE = find_spec("apex") is not None
DEEPSPEED_AVAILABLE = find_spec("deepspeed") is not None

#: Backend request shared by Models and dispatched operators.
Backend = Literal["auto", "eager", "sdpa", "oops", "cueq", "apex", "deepspeed"]
#: All public backend requests.
PUBLIC_BACKENDS = frozenset(get_args(Backend))


def resolve_backend(
    requested: str,
    *,
    supported: tuple[str, ...],
    public_default: str,
    oops: Callable[..., object] | None = None,
) -> str:
    r"""
    Resolve a backend request without tensor-specific constraints.

    ``"auto"`` selects the first installed implementation in ``supported``; OOps
    counts as installed only when it provides ``oops``, the operator its
    implementation calls. An explicit ``"oops"`` that ``supported`` lists requires
    that operator and never resolves to another backend. Other explicit providers
    remain operator preferences that retain their lazy dependency checks.

    Args:
        requested: The requested backend.
        supported: The backends this particular operator implements, in preference order.
        public_default: The operator's default PyTorch implementation.
        oops: The OOps kernel (``oops_kernel``) the ``"oops"`` implementation calls,
            whenever ``supported`` lists ``"oops"``.

    Returns:
        A concrete backend name drawn from ``supported`` (or ``public_default``).

    Raises:
        InvalidRequestError: ``oops_operator_unavailable`` for an explicit
            ``"oops"`` whose operator this installation does not provide.
        ValueError: If the request is not a public backend or a supported
            operator-specific implementation.
    """

    if requested == "auto":
        for backend in supported:
            if (
                (backend == "oops" and isinstance(oops, MissingOopsKernel))
                or (backend == "cueq" and not CUEQ_AVAILABLE)
                or (backend == "apex" and not APEX_AVAILABLE)
                or (backend == "deepspeed" and not DEEPSPEED_AVAILABLE)
            ):
                continue
            return backend
        return public_default
    if requested == "oops" and "oops" in supported and isinstance(oops, MissingOopsKernel):
        fail("oops_operator_unavailable", RequestRef("backend", requested), operator=oops.name)
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
    oops: Callable[..., object] | None = None,
) -> str:
    r"""
    Resolve a backend for concrete tensors.

    ``"auto"`` selects the first installed implementation in ``supported`` when
    every provided tensor is on CUDA. It otherwise selects ``public_default``.
    ``None`` values are ignored and at least one tensor is required for automatic
    accelerated selection. Installation follows ``resolve_backend``.

    Args:
        requested: The requested backend.
        tensors: Tensors considered by ``"auto"`` backend selection. ``None`` values are ignored so
            optional masks can be passed directly.
        supported: The backends this particular operator implements, in preference order.
        public_default: The operator's default PyTorch implementation.
        oops: The OOps kernel the ``"oops"`` implementation calls, whenever ``supported`` lists ``"oops"``.

    Returns:
        A concrete backend name drawn from ``supported`` (or ``public_default``).

    Raises:
        InvalidRequestError: ``oops_operator_unavailable`` for an explicit ``"oops"`` whose operator this
            installation does not provide.
    """

    if requested == "auto":
        has_tensor = False
        for tensor in tensors:
            if tensor is None:
                continue
            has_tensor = True
            if not tensor.is_cuda:
                return public_default
        if not has_tensor:
            return public_default
    return resolve_backend(requested, supported=supported, public_default=public_default, oops=oops)
