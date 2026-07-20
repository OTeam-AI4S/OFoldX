r"""Private backend selection for the four RFdiffusion operator contracts."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from . import _oops
from . import rfdiffusion as _torch_reference


def _load_oops_kernel(name: str) -> Callable[..., object] | None:
    if not _oops.OOPS_AVAILABLE:
        return None
    try:
        return _oops.oops_kernel(name)
    except AttributeError:
        return None


_oops_se3_basis = _load_oops_kernel("se3_basis")
_oops_segment_softmax_csr = _load_oops_kernel("segment_softmax_csr")
_oops_segment_sum_csr = _load_oops_kernel("segment_sum_csr")
_oops_weighted_tensor_product = _load_oops_kernel("weighted_tensor_product")


def se3_basis(
    relative_pos: Tensor,
    max_degree: int,
    *,
    backend: str = "auto",
) -> dict[str, Tensor]:
    r"""Dispatch the exact RFdiffusion SE(3)-basis contract."""

    selected = _select_backend(backend, relative_pos)
    if selected == "torch":
        return _torch_reference.se3_basis(relative_pos, max_degree)
    return _oops_se3_basis(  # type: ignore[misc,return-value]
        relative_pos,
        max_degree=max_degree,
        use_pad_trick=False,
        fully_fused=False,
    )


def segment_sum_csr(
    values: Tensor,
    row_offsets: Tensor,
    *,
    num_segments: int | None = None,
    backend: str = "auto",
) -> Tensor:
    r"""Dispatch the exact RFdiffusion CSR-sum contract."""

    selected = _select_backend(backend, values, row_offsets)
    if selected == "torch":
        return _torch_reference.segment_sum_csr(values, row_offsets, num_segments=num_segments)
    return _oops_segment_sum_csr(  # type: ignore[misc,return-value]
        values,
        row_offsets,
        num_segments=num_segments,
    )


def segment_softmax_csr(
    logits: Tensor,
    row_offsets: Tensor,
    *,
    backend: str = "auto",
) -> Tensor:
    r"""Dispatch the exact RFdiffusion CSR-softmax contract."""

    selected = _select_backend(backend, logits, row_offsets)
    if selected == "torch":
        return _torch_reference.segment_softmax_csr(logits, row_offsets)
    return _oops_segment_softmax_csr(logits, row_offsets)  # type: ignore[misc,return-value]


def weighted_tensor_product(
    features: Tensor,
    basis: Tensor,
    radial_weights: Tensor,
    *,
    backend: str = "auto",
) -> Tensor:
    r"""Dispatch the exact RFdiffusion weighted tensor-product contract."""

    selected = _select_backend(backend, features, basis, radial_weights)
    if selected == "torch":
        return _torch_reference.weighted_tensor_product(features, basis, radial_weights)
    return _oops_weighted_tensor_product(  # type: ignore[misc,return-value]
        features,
        basis,
        radial_weights,
    )


def _select_backend(backend: str, *tensors: Tensor) -> str:
    if backend not in {"auto", "torch", "oops"}:
        raise ValueError("backend must be exactly one of: auto, torch, oops")
    if backend == "auto":
        if _oops_adapter_available() and tensors and all(tensor.is_cuda for tensor in tensors):
            return "oops"
        return "torch"
    if backend == "oops" and not _oops_adapter_available():
        raise _oops.OOpsUnsupportedError("backend='oops' requires an optional OOps adapter with all four RF kernels")
    return backend


def _oops_adapter_available() -> bool:
    return _oops.OOPS_AVAILABLE and all(
        kernel is not None
        for kernel in (
            _oops_se3_basis,
            _oops_segment_softmax_csr,
            _oops_segment_sum_csr,
            _oops_weighted_tensor_product,
        )
    )
