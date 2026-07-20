r"""Public PyTorch reference operators required by the RFdiffusion SE(3) path.

These functions define correctness independently of the optional OOps package.  Their
contracts are intentionally the concrete contracts used by RFdiffusion rather than a
generic sparse/equivariant operator framework.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def se3_basis(relative_pos: Tensor, max_degree: int) -> dict[str, Tensor]:
    r"""Build the unpadded real SE(3) basis used by RFdiffusion.

    Args:
        relative_pos: Floating edge displacement tensor ``[E, 3]``.  Zero
            displacements are permitted and produce finite degree-zero-only
            harmonics.
        max_degree: Maximum representation degree.  RFdiffusion uses ``0`` or
            ``1``; no higher degree is part of this contract.

    Returns:
        Pair bases named ``"d_in,d_out"`` and the packed ``out{d}_fused`` and
        ``in{d}_fused`` bases consumed by the RFdiffusion convolutions.  For
        ``max_degree=1`` the pair shapes are ``[E,1,1,1]``, ``[E,1,1,3]``,
        ``[E,3,1,1]``, and ``[E,3,3,3]``.  Packed shapes are ``[E,4,2,1]``,
        ``[E,4,4,3]``, ``[E,1,2,4]``, and ``[E,3,4,4]``.

    The output preserves input device and dtype.  Float16/bfloat16 inputs are
    evaluated in float32 and cast back.  All tensor operations are ordinary
    PyTorch operations with autograd and FakeTensor support.  There is no mask
    argument: callers exclude masked graph edges before constructing
    ``relative_pos``.

    Raises:
        ValueError: If ``max_degree`` is outside ``{0, 1}``.
    """

    if max_degree not in (0, 1):
        raise ValueError("max_degree must be 0 or 1 for the RFdiffusion SE(3) contract")
    output_dtype = relative_pos.dtype
    values = relative_pos.float() if output_dtype in _LOW_PRECISION_DTYPES else relative_pos
    unit = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = unit.unbind(dim=-1)

    h0 = values.new_full((relative_pos.shape[0], 1), 0.28209479177387814)
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
    basis11 = h0.new_zeros((relative_pos.shape[0], 3, 3, 3))

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


def segment_sum_csr(values: Tensor, row_offsets: Tensor, *, num_segments: int | None = None) -> Tensor:
    r"""Sum edge-leading ``values`` inside destination-sorted CSR segments.

    ``values`` has shape ``[E, ...]`` and ``row_offsets`` has shape ``[N+1]``.
    Offsets are int32/int64 on the same device, start at zero, are monotonically
    non-decreasing, and end at ``E``.  The result has shape ``[N, ...]``;
    repeated offsets produce zero rows.  ``num_segments`` may append trailing
    zero rows but may not remove encoded rows.

    Floating dtype and device are preserved.  The operation is differentiable
    with respect to ``values`` only; offsets are discrete identity.  There is no
    mask argument: omitted edges do not appear in ``values`` or the CSR counts.

    """

    inferred_segments = row_offsets.shape[0] - 1
    output_segments = inferred_segments if num_segments is None else num_segments
    reduced = torch.segment_reduce(values, reduce="sum", offsets=row_offsets, axis=0)
    if output_segments == inferred_segments:
        return reduced
    trailing = reduced.new_zeros((output_segments - inferred_segments, *values.shape[1:]))
    return torch.cat((reduced, trailing), dim=0)


def segment_softmax_csr(logits: Tensor, row_offsets: Tensor) -> Tensor:
    r"""Apply a numerically stable softmax within each CSR segment.

    ``logits`` has shape ``[E, ...]`` and uses the same offset contract as
    :func:`segment_sum_csr`.  Softmax is taken over the edge axis independently
    for every trailing element.  Empty segments own no output entries.  Finite
    float16/bfloat16 logits use float32 max/subtract/exp/sum accumulation and the
    result is cast back to the input dtype.  Non-finite inputs follow ordinary
    PyTorch softmax propagation.  There is no mask argument: masked edges are
    omitted from the CSR representation.

    """

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


def weighted_tensor_product(features: Tensor, basis: Tensor, radial_weights: Tensor) -> Tensor:
    r"""Apply the RFdiffusion edge-wise basis and radial channel mixing.

    Args:
        features: Floating tensor ``[E, C_in, D_in]``.
        basis: Floating tensor ``[E, D_in, F, D_out]``.
        radial_weights: Floating tensor ``[E, C_out, C_in * F]``.

    Returns:
        Tensor ``[E, C_out, D_out]`` satisfying
        ``out[e,o,d] = sum(c,i,f, features[e,c,i] * basis[e,i,f,d] *
        radial_weights[e,o,c*F+f])``.

    All inputs must share floating dtype, device, and edge dimension.  Ordinary
    batched matrix multiplication supplies gradients for all three inputs.
    There is no mask argument: the operator evaluates exactly the supplied
    edges, including the well-defined empty ``E=0`` case.

    """

    edges, channels_in, dim_in = features.shape
    num_basis, dim_out = basis.shape[-2:]
    projected = torch.matmul(features, basis.reshape(edges, dim_in, num_basis * dim_out))
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
