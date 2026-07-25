r"""
Backend-aware graph and SE(3) operators.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from ._oops import oops_kernel
from .resolve import resolve_config_backend, resolve_tensor_backend

_oops_se3_basis = oops_kernel("se3_basis")
_oops_segment_softmax_csr = oops_kernel("segment_softmax_csr")
_oops_segment_sum_csr = oops_kernel("segment_sum_csr")
_oops_weighted_tensor_product = oops_kernel("weighted_tensor_product")

_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED = ("oops", "eager")


class Se3Basis(nn.Module):
    r"""
    Build the real SE(3) basis used by RFdiffusion convolutions.

    Args:
        max_degree: Maximum representation degree. RFdiffusion supports zero or one.
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    backend: str

    def __init__(self, max_degree: int, *, backend: str = "auto") -> None:
        super().__init__()
        self.max_degree = max_degree
        self.backend = resolve_config_backend(
            backend,
            supported=_SUPPORTED,
            public_default="eager",
        )

    def forward(self, relative_positions: Tensor) -> dict[str, Tensor]:
        r"""
        Construct pair and fused bases for edge displacement vectors.

        Args:
            relative_positions: Edge displacement vectors with shape ``(E, 3)``.

        Returns:
            Pair bases and their fused degree-zero/degree-one layouts.
        """

        backend = resolve_tensor_backend(
            self.backend,
            tensors=(relative_positions,),
            supported=_SUPPORTED,
            public_default="eager",
        )
        if backend == "oops":
            return cast(
                dict[str, Tensor],
                _oops_se3_basis(
                    relative_positions,
                    max_degree=self.max_degree,
                    use_pad_trick=False,
                    fully_fused=False,
                ),
            )
        return _se3_basis(relative_positions, self.max_degree)


class SegmentSumCsr(nn.Module):
    r"""
    Sum edge-leading values within destination-sorted CSR segments.

    Args:
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    backend: str

    def __init__(self, *, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(
            backend,
            supported=_SUPPORTED,
            public_default="eager",
        )

    def forward(
        self,
        values: Tensor,
        row_offsets: Tensor,
        *,
        num_segments: int | None = None,
    ) -> Tensor:
        r"""
        Reduce one CSR edge tensor.

        Args:
            values: Edge-leading values with shape ``(E, ...)``.
            row_offsets: CSR offsets with shape ``(N + 1,)``.
            num_segments: Optional output row count including trailing empty rows.

        Returns:
            Segment sums with shape ``(N, ...)``.
        """

        backend = resolve_tensor_backend(
            self.backend,
            tensors=(values, row_offsets),
            supported=_SUPPORTED,
            public_default="eager",
        )
        if backend == "oops":
            return cast(
                Tensor,
                _oops_segment_sum_csr(
                    values,
                    row_offsets,
                    num_segments=num_segments,
                ),
            )
        return _segment_sum_csr(values, row_offsets, num_segments=num_segments)


class SegmentSoftmaxCsr(nn.Module):
    r"""
    Apply a numerically stable softmax within CSR segments.

    Args:
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    backend: str

    def __init__(self, *, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(
            backend,
            supported=_SUPPORTED,
            public_default="eager",
        )

    def forward(self, logits: Tensor, row_offsets: Tensor) -> Tensor:
        r"""
        Normalize one CSR edge tensor.

        Args:
            logits: Edge-leading logits with shape ``(E, ...)``.
            row_offsets: CSR offsets with shape ``(N + 1,)``.

        Returns:
            Per-segment softmax values with the same shape and dtype as ``logits``.
        """

        backend = resolve_tensor_backend(
            self.backend,
            tensors=(logits, row_offsets),
            supported=_SUPPORTED,
            public_default="eager",
        )
        if backend == "oops":
            return cast(Tensor, _oops_segment_softmax_csr(logits, row_offsets))
        return _segment_softmax_csr(logits, row_offsets)


class WeightedTensorProduct(nn.Module):
    r"""
    Apply edge-wise basis projection and radial channel mixing.

    Args:
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    backend: str

    def __init__(self, *, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(
            backend,
            supported=_SUPPORTED,
            public_default="eager",
        )

    def forward(
        self,
        features: Tensor,
        basis: Tensor,
        radial_weights: Tensor,
    ) -> Tensor:
        r"""
        Mix representation components and learned radial channels.

        Args:
            features: Input features with shape ``(E, C_in, D_in)``.
            basis: Basis values with shape ``(E, D_in, F, D_out)``.
            radial_weights: Radial weights with shape ``(E, C_out, C_in * F)``.

        Returns:
            Output features with shape ``(E, C_out, D_out)``.
        """

        backend = resolve_tensor_backend(
            self.backend,
            tensors=(features, basis, radial_weights),
            supported=_SUPPORTED,
            public_default="eager",
        )
        if backend == "oops":
            return cast(
                Tensor,
                _oops_weighted_tensor_product(
                    features,
                    basis,
                    radial_weights,
                ),
            )
        return _weighted_tensor_product(features, basis, radial_weights)


def _se3_basis(relative_positions: Tensor, max_degree: int) -> dict[str, Tensor]:
    if max_degree not in (0, 1):
        raise ValueError("max_degree must be 0 or 1 for the RFdiffusion SE(3) contract")
    output_dtype = relative_positions.dtype
    values = relative_positions.float() if output_dtype in _LOW_PRECISION_DTYPES else relative_positions
    unit = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = unit.unbind(dim=-1)

    h0 = values.new_full((relative_positions.shape[0], 1), 0.28209479177387814)
    basis00 = h0.reshape(-1, 1, 1, 1)
    if max_degree == 0:
        basis00 = basis00.to(dtype=output_dtype)
        return {
            "0,0": basis00,
            "out0_fused": basis00,
            "in0_fused": basis00,
        }

    h1 = torch.stack(
        (
            0.4886025119029199 * x,
            0.4886025119029199 * y,
            0.4886025119029199 * z,
        ),
        dim=-1,
    )
    h2 = torch.stack(
        (
            1.0925484305920792 * x * z,
            1.0925484305920792 * x * y,
            0.31539156525252005 * (2.0 * y * y - x * x - z * z),
            1.0925484305920792 * y * z,
            0.5462742152960396 * (z * z - x * x),
        ),
        dim=-1,
    )

    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    inv_sqrt6 = 1.0 / math.sqrt(6.0)
    inv_sqrt10 = 1.0 / math.sqrt(10.0)
    inv_sqrt30 = 1.0 / math.sqrt(30.0)
    basis01 = (h1 * inv_sqrt3).reshape(-1, 1, 1, 3)
    basis10 = (h1 * inv_sqrt3).reshape(-1, 3, 1, 1)
    basis11 = h0.new_zeros((relative_positions.shape[0], 3, 3, 3))

    j0 = h0[:, 0] * inv_sqrt3
    basis11[:, 0, 0, 0] = j0
    basis11[:, 1, 0, 1] = j0
    basis11[:, 2, 0, 2] = j0
    basis11[:, 1, 1, 0] = -h1[:, 2] * inv_sqrt6
    basis11[:, 2, 1, 0] = h1[:, 1] * inv_sqrt6
    basis11[:, 0, 1, 1] = h1[:, 2] * inv_sqrt6
    basis11[:, 2, 1, 1] = -h1[:, 0] * inv_sqrt6
    basis11[:, 0, 1, 2] = -h1[:, 1] * inv_sqrt6
    basis11[:, 1, 1, 2] = h1[:, 0] * inv_sqrt6
    basis11[:, 0, 2, 0] = -h2[:, 2] * inv_sqrt30 - h2[:, 4] * inv_sqrt10
    basis11[:, 1, 2, 0] = h2[:, 1] * inv_sqrt10
    basis11[:, 2, 2, 0] = h2[:, 0] * inv_sqrt10
    basis11[:, 0, 2, 1] = h2[:, 1] * inv_sqrt10
    basis11[:, 1, 2, 1] = 2.0 * h2[:, 2] * inv_sqrt30
    basis11[:, 2, 2, 1] = h2[:, 3] * inv_sqrt10
    basis11[:, 0, 2, 2] = h2[:, 0] * inv_sqrt10
    basis11[:, 1, 2, 2] = h2[:, 3] * inv_sqrt10
    basis11[:, 2, 2, 2] = -h2[:, 2] * inv_sqrt30 + h2[:, 4] * inv_sqrt10

    basis = {
        "0,0": basis00.to(dtype=output_dtype),
        "0,1": basis01.to(dtype=output_dtype),
        "1,0": basis10.to(dtype=output_dtype),
        "1,1": basis11.to(dtype=output_dtype),
    }
    basis.update(_fused_degree_one_bases(basis))
    return basis


def _segment_sum_csr(
    values: Tensor,
    row_offsets: Tensor,
    *,
    num_segments: int | None = None,
) -> Tensor:
    inferred_segments = row_offsets.shape[0] - 1
    output_segments = inferred_segments if num_segments is None else num_segments
    reduced = torch.segment_reduce(values, reduce="sum", offsets=row_offsets, axis=0)
    if output_segments == inferred_segments:
        return reduced
    trailing = reduced.new_zeros((output_segments - inferred_segments, *values.shape[1:]))
    return torch.cat((reduced, trailing), dim=0)


def _segment_softmax_csr(logits: Tensor, row_offsets: Tensor) -> Tensor:
    inferred_segments = row_offsets.shape[0] - 1
    work = logits.float() if logits.dtype in _LOW_PRECISION_DTYPES else logits
    lengths = row_offsets[1:] - row_offsets[:-1]
    segment_ids = torch.repeat_interleave(
        torch.arange(inferred_segments, device=logits.device, dtype=torch.long),
        lengths,
        output_size=logits.shape[0],
    )
    maxima = torch.segment_reduce(work, reduce="max", offsets=row_offsets, axis=0)
    shifted = work - maxima.index_select(0, segment_ids)
    exponentials = shifted.exp()
    denominators = torch.segment_reduce(exponentials, reduce="sum", offsets=row_offsets, axis=0)
    output = exponentials / denominators.index_select(0, segment_ids)
    return output.to(dtype=logits.dtype) if output.dtype != logits.dtype else output


def _weighted_tensor_product(
    features: Tensor,
    basis: Tensor,
    radial_weights: Tensor,
) -> Tensor:
    edges, channels_in, dim_in = features.shape
    num_basis, dim_out = basis.shape[-2:]
    projected = torch.matmul(
        features,
        basis.reshape(edges, dim_in, num_basis * dim_out),
    )
    projected = projected.reshape(edges, channels_in * num_basis, dim_out)
    return torch.matmul(radial_weights, projected)


def _fused_degree_one_bases(basis: dict[str, Tensor]) -> dict[str, Tensor]:
    basis00, basis01, basis10, basis11 = (basis[key] for key in ("0,0", "0,1", "1,0", "1,1"))
    edges = basis00.shape[0]
    out0 = basis00.new_zeros((edges, 4, 2, 1))
    out0[:, 0:1, 0:1] = basis00
    out0[:, 1:4, 1:2] = basis10
    out1 = basis00.new_zeros((edges, 4, 4, 3))
    out1[:, 0:1, 0:1] = basis01
    out1[:, 1:4, 1:4] = basis11
    in0 = basis00.new_zeros((edges, 1, 2, 4))
    in0[:, :, 0:1, 0:1] = basis00
    in0[:, :, 1:2, 1:4] = basis01
    in1 = basis00.new_zeros((edges, 3, 4, 4))
    in1[:, :, 0:1, 0:1] = basis10
    in1[:, :, 1:4, 1:4] = basis11
    return {
        "out0_fused": out0,
        "out1_fused": out1,
        "in0_fused": in0,
        "in1_fused": in1,
    }
