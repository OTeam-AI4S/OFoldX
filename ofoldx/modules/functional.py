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
Functional tensor helpers for modules.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor
from torch.autograd.function import once_differentiable

from .._oops import oops_kernel
from .backend.resolve import resolve_tensor_backend

_oops_pair_mask_from_token_mask = oops_kernel("pair_mask_from_token_mask")
_oops_pair_mask_from_msa_mask = oops_kernel("pair_mask_from_msa_mask")
_oops_weighted_mean = oops_kernel("weighted_mean")


def project_categories(values: Tensor, available: Tensor, *, num_classes: int) -> Tensor:
    """Select available categorical IDs before validating or converting them.

    The Boolean mask broadcasts to the existing value axes. Excluded payload
    has safe ID zero; this fill asserts no observed class. Available IDs must
    be finite exact integers in ``[0, num_classes)``. The result is int64.
    """
    if num_classes <= 0:
        raise ValueError("a categorical vocabulary must be nonempty")
    if available.dtype != torch.bool or available.device != values.device:
        raise TypeError("categorical availability must be Boolean on the value device")
    if values.is_complex():
        raise TypeError("categorical IDs must be real integers")
    try:
        shape = torch.broadcast_shapes(values.shape, available.shape)
    except RuntimeError as error:
        raise ValueError("categorical availability must match the value axes") from error
    if shape != values.shape:
        raise ValueError("categorical availability cannot introduce value axes")
    selected = torch.where(available, values, 0)
    raw = selected.concat if isinstance(selected, NestedTensor) else selected
    valid = ((raw >= 0) & (raw < num_classes)).all()
    if raw.is_floating_point():
        valid = valid & torch.isfinite(raw).all() & (raw == raw.trunc()).all()
    if torch.compiler.is_compiling():
        torch._assert_async(valid, "available categorical IDs must be exact vocabulary indices")
    elif not bool(valid):
        raise ValueError("available categorical IDs must be exact vocabulary indices")
    return selected.long()


def project_counts(values: Tensor, available: Tensor) -> Tensor:
    """Select raw nonnegative integer counts without narrowing their dtype.

    Float carriers must contain finite exact integers on available rows. This
    does not apply to transformed deletion features or continuous averages.
    """
    if available.dtype != torch.bool or available.device != values.device:
        raise TypeError("count availability must be Boolean on the value device")
    if values.is_complex():
        raise TypeError("counts must be real nonnegative integers")
    try:
        shape = torch.broadcast_shapes(values.shape, available.shape)
    except RuntimeError as error:
        raise ValueError("count availability must match the value axes") from error
    if shape != values.shape:
        raise ValueError("count availability cannot introduce value axes")
    selected = torch.where(available, values, 0)
    raw = selected.concat if isinstance(selected, NestedTensor) else selected
    valid = (raw >= 0).all()
    if raw.is_floating_point():
        valid = valid & torch.isfinite(raw).all() & (raw == raw.trunc()).all()
    if torch.compiler.is_compiling():
        torch._assert_async(valid, "available counts must be finite nonnegative integers")
    elif not bool(valid):
        raise ValueError("available counts must be finite nonnegative integers")
    return selected


def _scaled_power(value: Tensor, exponent: Tensor) -> Tensor:
    r"""Apply a binary exponent without overflowing an intermediate power of two."""
    exponent = torch.where(value == 0, 0, exponent)
    first = exponent.clamp(-1022, 1023)
    second = (exponent - first).clamp(-1022, 1023)
    return torch.ldexp(torch.ldexp(torch.ldexp(value, first), second), exponent - first - second)


class _WeightedMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: Tensor, weights: Tensor, row_groups: Tensor | None, num_groups: int) -> Tensor:
        def reduce_sum(value: Tensor) -> Tensor:
            if row_groups is None:
                return value.sum(-1, keepdim=True)
            output = value.new_zeros((num_groups, *value.shape[1:]))
            return output.index_add(0, row_groups, value)

        def reduce_max(value: Tensor) -> Tensor:
            if row_groups is None:
                identity = value.new_full((*value.shape[:-1], 1), -4096)
                return torch.cat((value, identity), -1).amax(-1, keepdim=True)
            output = value.new_full((num_groups, *value.shape[1:]), -4096)
            index = row_groups.reshape(-1, *((1,) * (value.dim() - 1))).expand_as(value)
            return output.scatter_reduce(0, index, value, "amax", include_self=True)

        def expand_groups(value: Tensor) -> Tensor:
            return value if row_groups is None else value.index_select(0, row_groups)

        x, w = values.double(), weights.double()
        x = torch.where((w == 0) & ~torch.isfinite(x), 0.0, x)
        ctx.exponents = values.dtype == torch.float64 or weights.dtype == torch.float64
        if ctx.exponents:
            xm, xe = torch.frexp(x)
            wm, we = torch.frexp(w)
            de = reduce_max(torch.where(w != 0, we, -4096))
            ne = reduce_max(torch.where((x != 0) & (w != 0), xe + we, -4096))
            denominator = reduce_sum(_scaled_power(wm, we - expand_groups(de)))
            numerator = reduce_sum(_scaled_power(xm * wm, xe + we - expand_groups(ne)))
            nm, nen = torch.frexp(numerator)
            dm, den = torch.frexp(denominator)
            mm = nm / torch.where(denominator > 0, dm, 1.0)
            me = ne - de + nen - den
            mean = _scaled_power(mm, me)
        else:
            denominator = reduce_sum(w)
            de = torch.zeros_like(denominator, dtype=torch.int32)
            mean = reduce_sum(x * w) / torch.where(denominator > 0, denominator, 1.0)
            mm, me = torch.frexp(mean)
        present = denominator > 0
        mm = torch.where(present, mm, 0.0)
        ctx.save_for_backward(values, weights, mm, me, denominator, de, row_groups)
        dtype = torch.promote_types(values.dtype, weights.dtype) if weights.is_floating_point() else values.dtype
        result = torch.where(present, mean, float("nan")).to(dtype)
        return result.squeeze(-1) if row_groups is None else result

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, Tensor | None, None, None]:
        values, weights, mm, me, denominator, de, row_groups = ctx.saved_tensors
        x, w = values.double(), weights.double()
        if row_groups is None:
            g = gradient.double().unsqueeze(-1)
        else:
            g = gradient.double().index_select(0, row_groups)
            mm, me, denominator, de = (value.index_select(0, row_groups) for value in (mm, me, denominator, de))
        x = torch.where((w == 0) & ~torch.isfinite(x), 0.0, x)
        present = denominator > 0
        denominator = torch.where(present, denominator, 1.0)
        if ctx.exponents:
            gm, ge = torch.frexp(g)
            wm, we = torch.frexp(w)
            gx = _scaled_power(gm * wm / denominator, ge + we - de)
            xm, xe = torch.frexp(x)
            delta_e = torch.maximum(torch.where(x != 0, xe, -4096), torch.where(mm != 0, me, -4096))
            delta = _scaled_power(xm, xe - delta_e) - _scaled_power(mm.expand_as(xm), me - delta_e)
            gw = _scaled_power(gm * delta / denominator, ge + delta_e - de)
        else:
            mean = _scaled_power(mm, me)
            gx, gw = g * w / denominator, g * (x - mean) / denominator
        gx = torch.where(present & (w != 0), gx, 0.0).to(values.dtype)
        weight_gradient = torch.where(present, gw, 0.0).to(weights.dtype) if weights.is_floating_point() else None
        return gx, weight_gradient, None, None


def weighted_mean(values: Tensor, weights: Tensor, *, backend: str = "auto") -> Tensor:
    r"""Reduce the final observation axis with nonnegative broadcast coefficients.

    Floating operands follow ordinary promotion; integer masks retain the value
    dtype. Finite zero-weight observations retain coefficient sensitivity;
    nonfinite zero-weight observations are treated as zero. An absent domain
    reports NaN with zero gradients. First-order coupled adjoints retain the
    caller's storage dtype. Nested samples use their true observation lengths
    and retain gradients to their original tensor storage. The eager backend
    lowers static final axes, sole ragged final axes and final ragged pair
    reductions through packed tensors, including static sample prefixes and
    fullgraph compilation.
    """
    resolved = resolve_tensor_backend(
        backend,
        tensors=(values, weights),
        supported=("oops", "eager"),
        public_default="eager",
        oops=_oops_weighted_mean,
    )
    if isinstance(values, NestedTensor) or isinstance(weights, NestedTensor):
        layout = values if isinstance(values, NestedTensor) else cast(NestedTensor, weights)
        if not isinstance(values, NestedTensor):
            values = layout.nested_like(torch.broadcast_to(values, layout.shape), strict=False)
        if not isinstance(weights, NestedTensor):
            weights = layout.nested_like(torch.broadcast_to(weights, layout.shape), strict=False)
        values, weights = torch.broadcast_tensors(values, weights)
        layout = cast(NestedTensor, values)
        coefficients = cast(NestedTensor, weights)
        rank = layout.dim() - 1
        ragged = layout.ragged_dims
        order = layout.packed_dim_order
        if (
            resolved == "eager"
            and ragged
            and ragged == tuple(range(len(ragged)))
            and order == tuple(range(rank))
            and rank - 1 not in ragged
        ):
            result = weighted_mean(layout.concat, coefficients.concat, backend=resolved)
            return layout.packed_with_static_tail(result)
        if resolved == "eager" and len(ragged) == 1 and ragged[0] == rank - 1 and order == (rank - 1, *range(rank - 1)):
            groups = layout.packed_batch_indices(device=layout.device)
            result = cast(Tensor, _WeightedMean.apply(layout.concat, coefficients.concat, groups, len(layout)))
            return result if layout.batch_first else result.movedim(0, min(1, result.dim() - 1))
        if (
            resolved == "eager"
            and len(ragged) == 2
            and ragged == (rank - 2, rank - 1)
            and order == (*ragged, *range(rank - 2))
            and (layout.batch_first or rank == 2)
        ):
            offsets = layout.ragged_level_offsets(-1, device=layout.device, dtype=torch.long)
            count = offsets.numel() - 1
            groups = torch.repeat_interleave(
                torch.arange(count, device=layout.device),
                offsets[1:] - offsets[:-1],
                output_size=layout.concat.shape[0],
            )
            result = cast(Tensor, _WeightedMean.apply(layout.concat, coefficients.concat, groups, count))
            output = layout.packed_with_lengths(result, layout.element_sizes()[:, ragged[0]])
            return output.movedim(1, ragged[0] + 1) if layout.batch_first and ragged[0] else output
        results = [
            weighted_mean(sample, coefficient, backend=resolved)
            for sample, coefficient in zip(
                layout.unbind(0 if layout.batch_first else 1),
                coefficients.unbind(0 if coefficients.batch_first else 1),
                strict=True,
            )
        ]
        if not results:
            dtype = torch.promote_types(values.dtype, weights.dtype) if weights.is_floating_point() else values.dtype
            return torch.empty(layout.shape[:-1], dtype=dtype, device=layout.device)
        if all(result.shape == results[0].shape for result in results):
            return torch.stack(results, dim=0 if layout.batch_first else min(1, results[0].ndim))
        return NestedTensor(results, batch_first=layout.batch_first)
    if resolved == "oops":
        return cast(Tensor, _oops_weighted_mean(values, weights))
    values, weights = torch.broadcast_tensors(values, weights)
    return cast(Tensor, _WeightedMean.apply(values, weights, None, 0))


def indexed_sum(values: Tensor, index: Tensor, num_groups: int, dim: int = 0) -> Tensor:
    r"""Sum slices along ``dim`` into the groups identified by ``index``.

    Args:
        values: Values whose slices along ``dim`` are summed independently.
        index: One-dimensional group index for every slice along ``dim``.
        num_groups: Number of output groups, including empty groups.
        dim: Axis of the input slices and the output groups.

    Returns:
        Sums in the input dtype and device, with ``num_groups`` entries along
        ``dim``. Empty groups are zero.
    """

    leading = values.movedim(dim, 0)
    output = values.new_zeros((num_groups, *leading.shape[1:]))
    output.index_add_(0, index, leading)
    return output.movedim(0, dim)


def grouped_softmax(values: Tensor, index: Tensor, num_groups: int, dim: int = 0) -> Tensor:
    r"""Normalize slices along ``dim`` within their indexed groups.

    Args:
        values: Logits whose slices along ``dim`` are normalized independently.
        index: One-dimensional group index for every slice along ``dim``.
        num_groups: Number of groups, including empty groups.
        dim: Axis of the slices grouped by ``index``.

    Returns:
        Probabilities with the input shape, dtype and device. Every nonempty
        group with at least one finite logit sums to one for each remaining
        coordinate. Empty groups have no output entries.
    """

    leading = values.movedim(dim, 0)
    expanded = index.reshape(index.shape[0], *((1,) * (leading.ndim - 1))).expand_as(leading)
    maxima = values.new_full((num_groups, *leading.shape[1:]), -torch.inf)
    maxima.scatter_reduce_(0, expanded, leading, reduce="amax", include_self=True)
    exponentials = (leading - maxima.index_select(0, index)).exp()
    denominators = indexed_sum(exponentials, index, num_groups)
    return (exponentials / denominators.index_select(0, index)).movedim(0, dim)


@torch.library.custom_op("ofoldx::_packed_pair_mask", mutates_args=())
def _packed_pair_mask(token_values: Tensor, token_offsets: Tensor) -> Tensor:
    r"""Build packed square pair values from packed tokens and device row splits."""

    lengths = token_offsets[1:] - token_offsets[:-1]
    batch_indices = torch.arange(lengths.shape[0], device=token_offsets.device, dtype=torch.long)
    token_batch_indices = torch.repeat_interleave(batch_indices, lengths)
    row_values = torch.repeat_interleave(token_values, lengths.index_select(0, token_batch_indices))
    pair_lengths = lengths.square()
    pair_batch_indices = torch.repeat_interleave(batch_indices, pair_lengths)
    pair_offsets = F.pad(pair_lengths.cumsum(0), (1, 0))
    positions = torch.arange(pair_batch_indices.shape[0], device=token_offsets.device, dtype=torch.long)
    local_positions = positions - pair_offsets.index_select(0, pair_batch_indices)
    columns = torch.remainder(local_positions, lengths.index_select(0, pair_batch_indices))
    column_indices = token_offsets.index_select(0, pair_batch_indices) + columns
    return row_values & token_values.index_select(0, column_indices)


@torch.library.register_fake(_packed_pair_mask)
def _packed_pair_mask_fake(token_values: Tensor, token_offsets: Tensor) -> Tensor:
    del token_offsets
    output_size = torch.library.get_ctx().new_dynamic_size(min=0)
    return token_values.new_empty((output_size,), dtype=torch.bool)


def pair_mask_from_token_mask(token_mask: Tensor | NestedTensor, *, backend: str = "auto") -> Tensor | NestedTensor:
    r"""
    Build a pair-validity mask by taking the outer conjunction of a token mask.

    A token pair ``(i, j)`` is valid only when both token ``i`` and token ``j`` are valid, so the
    result is ``token_mask[..., :, None] & token_mask[..., None, :]``.

    Args:
        token_mask: `[..., N_token]` token-validity mask; any truthy dtype is accepted.

    Returns:
        `[..., N_token, N_token]` boolean pair-validity mask.
    """

    if isinstance(token_mask, NestedTensor):
        if (
            not token_mask.batch_first
            or token_mask.dim() != 2
            or token_mask.ragged_dims != (0,)
            or token_mask.packed_dim_order != (0,)
        ):
            raise ValueError("nested token_mask elements must use explicit canonical (N_i,) packing")
        token_values = token_mask.concat.bool()
        if (
            resolve_tensor_backend(
                backend,
                tensors=(token_values,),
                supported=("oops", "eager"),
                public_default="eager",
                oops=_oops_pair_mask_from_token_mask,
            )
            == "oops"
        ):
            return cast(NestedTensor, _oops_pair_mask_from_token_mask(token_mask))
        device_offsets = token_mask.packed_offsets(device=token_values.device, dtype=torch.int64)
        pair_values = _packed_pair_mask(token_values, device_offsets)
        cpu_offsets = token_mask.packed_offsets(dtype=torch.int64)
        return token_mask.packed_with_square_lengths(pair_values, cpu_offsets[1:] - cpu_offsets[:-1])

    token_mask = token_mask.bool()
    return token_mask[..., :, None] & token_mask[..., None, :]


def banded_pair_mask(axis: Tensor | NestedTensor, half_window: int) -> Tensor | NestedTensor:
    r"""
    Build a pair mask keeping positions within ``half_window`` of each other along one axis.

    A pair ``(i, j)`` is kept when ``abs(i - j) <= half_window``, where ``i`` and ``j`` count
    positions inside their own batch element. The diagonal is always kept, so a window of ``0``
    leaves every position attending only to itself. ``axis`` supplies the axis topology and its
    values are not read: for a dense tensor every row is a real position, and for a
    ``NestedTensor`` each element's own length bounds its band.

    Args:
        axis: `[..., N]` tensor, or a nested tensor with canonical ``(N_i,)`` elements, whose
            trailing axis is the banded one.
        half_window: Largest kept separation in positions; must not be negative.

    Returns:
        `[..., N, N]` boolean pair mask, nested when ``axis`` is nested.

    Raises:
        ValueError: If ``half_window`` is negative, or a nested ``axis`` is not canonically packed.
    """

    if half_window < 0:
        raise ValueError(f"half_window must not be negative, got {half_window}")

    if isinstance(axis, NestedTensor):
        if not axis.batch_first or axis.dim() != 2 or axis.ragged_dims != (0,) or axis.packed_dim_order != (0,):
            raise ValueError("nested axis elements must use explicit canonical (N_i,) packing")
        cpu_offsets = axis.packed_offsets(dtype=torch.int64)
        lengths = cpu_offsets[1:] - cpu_offsets[:-1]
        device = axis.concat.device
        pair_values = _packed_banded_mask(
            axis.packed_offsets(device=device, dtype=torch.int64),
            half_window,
        )
        return axis.packed_with_square_lengths(pair_values, lengths)

    positions = torch.arange(axis.shape[-1], device=axis.device)
    return (positions[:, None] - positions[None, :]).abs() <= half_window


@torch.library.custom_op("ofoldx::_packed_banded_mask", mutates_args=())
def _packed_banded_mask(offsets: Tensor, half_window: int) -> Tensor:
    r"""Build packed square band values from device row splits."""

    lengths = offsets[1:] - offsets[:-1]
    batch_indices = torch.arange(lengths.shape[0], device=offsets.device, dtype=torch.long)
    pair_lengths = lengths.square()
    pair_batch_indices = torch.repeat_interleave(batch_indices, pair_lengths)
    pair_offsets = F.pad(pair_lengths.cumsum(0), (1, 0))
    positions = torch.arange(pair_batch_indices.shape[0], device=offsets.device, dtype=torch.long)
    local_positions = positions - pair_offsets.index_select(0, pair_batch_indices)
    pair_token_counts = lengths.index_select(0, pair_batch_indices)
    rows = torch.div(local_positions, pair_token_counts, rounding_mode="floor")
    columns = torch.remainder(local_positions, pair_token_counts)
    return (rows - columns).abs() <= half_window


@torch.library.register_fake(_packed_banded_mask)
def _packed_banded_mask_fake(offsets: Tensor, half_window: int) -> Tensor:
    del half_window
    output_size = torch.library.get_ctx().new_dynamic_size(min=0)
    return offsets.new_empty((output_size,), dtype=torch.bool)


def pair_mask_from_msa_mask(msa_mask: Tensor | NestedTensor, *, backend: str = "auto") -> Tensor | NestedTensor:
    r"""
    Build a token-pair mask from an MSA mask, keeping tokens with any valid MSA row.

    A token is treated as valid when at least one MSA row covers it (``any`` over the MSA axis);
    the per-token mask is then expanded into a pair mask via
    [pair_mask_from_token_mask][ofoldx.modules.functional.pair_mask_from_token_mask].

    Args:
        msa_mask: `[..., N_msa, N_token]` MSA-validity mask; any truthy dtype is accepted.

    Returns:
        `[..., N_token, N_token]` boolean pair-validity mask for tokens with at least one valid
            MSA row.
    """

    if (
        isinstance(msa_mask, NestedTensor)
        and resolve_tensor_backend(
            backend,
            tensors=(msa_mask,),
            supported=("oops", "eager"),
            public_default="eager",
            oops=_oops_pair_mask_from_msa_mask,
        )
        == "oops"
    ):
        return cast(NestedTensor, _oops_pair_mask_from_msa_mask(msa_mask))
    token_mask = msa_mask.bool().any(dim=-2)
    return pair_mask_from_token_mask(token_mask, backend=backend)


def pairwise_distance(coordinates: Tensor, other: Tensor | None = None) -> Tensor:
    r"""Return Euclidean distances over dense coordinate axes ``[..., N, D]``.

    The output is ``[..., N, M]``; omitted ``other`` compares the same axis.
    Float32/float64 use native ``cdist`` on coordinate differences, avoiding
    cancellation between squared terms under a common translation. Lower
    precision uses native vector norms without coordinate conversion.
    Coincident points have the ordinary norm zero subgradient.
    """

    if other is None:
        other = coordinates
    if coordinates.dtype in (torch.float16, torch.bfloat16):
        return torch.linalg.vector_norm(coordinates.unsqueeze(-2) - other.unsqueeze(-3), dim=-1)
    return torch.cdist(coordinates, other, compute_mode="donot_use_mm_for_euclid_dist")


def coordinate_distogram(
    positions: Tensor,
    *,
    min_distance: float,
    max_distance: float,
    num_bins: int,
) -> Tensor:
    r"""
    Encode pairwise coordinate distances in strict-open distance intervals.

    Lower edges are ``num_bins`` evenly spaced distances from ``min_distance`` to
    ``max_distance``, inclusive. Each upper edge is the next lower edge, except
    the last squared upper edge, which is ``1e8``. Distances equal to an edge or
    containing NaN belong to no interval.

    Args:
        positions: Coordinates ``[..., N, 3]``, in Angstrom.
        min_distance: First distance-bin lower edge, in Angstrom.
        max_distance: Last distance-bin lower edge, in Angstrom.
        num_bins: Number of distance intervals.

    Returns:
        Binary features ``[..., N, N, num_bins]`` in ``positions``' dtype and device.
    """

    squared_distances = (positions.unsqueeze(-2) - positions.unsqueeze(-3)).square().sum(dim=-1, keepdim=True)
    lower = torch.linspace(
        min_distance,
        max_distance,
        num_bins,
        device=positions.device,
        dtype=positions.dtype,
    ).square()
    upper = torch.cat((lower[1:], lower.new_tensor([1e8])))
    return ((squared_distances > lower) & (squared_distances < upper)).to(dtype=positions.dtype)


def gaussian_radial_basis(distances: Tensor, centers: Tensor, width: float | Tensor) -> Tensor:
    r"""Expand scalar distances as ``exp(-((distance - center) / width)**2)``.

    Centers occupy the output feature axis; width broadcasts with that axis.
    Coordinates, centers and width retain ordinary PyTorch dtype promotion.
    """

    return torch.exp(-(((distances.unsqueeze(-1) - centers) / width) ** 2))


def radial_basis(
    distances: Tensor,
    num_bins: int = 16,
    lower_bound: float = 2.0,
    upper_bound: float = 22.0,
    *,
    width: float | None = None,
) -> Tensor:
    r"""
    Expand distances over evenly spaced Gaussian radial-basis centers.

    Centers are ``num_bins`` points spanning ``[lower_bound, upper_bound]`` inclusive, and the shared
    default width is ``(upper_bound - lower_bound) / num_bins``. This is the range divided by
    the bin count, not the spacing between centers. An explicit ``width`` selects another
    Gaussian scale without changing the centers.

    Args:
        distances: Distances of any shape `[...]`, in Angstrom.
        num_bins: Number of radial-basis centers.
        lower_bound: First center, in Angstrom.
        upper_bound: Last center, in Angstrom.
        width: Gaussian width, in Angstrom. ``None`` uses the range divided by the bin count.

    Returns:
        Radial-basis features `[..., num_bins]`.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.functional import radial_basis
        >>> radial_basis(torch.tensor([2.0]), num_bins=4, lower_bound=2.0, upper_bound=10.0).shape
        torch.Size([1, 4])
        >>> float(radial_basis(torch.tensor([2.0]), num_bins=4, lower_bound=2.0, upper_bound=10.0)[0, 0])
        1.0
    """

    centers = torch.linspace(
        lower_bound,
        upper_bound,
        num_bins,
        device=distances.device,
        dtype=distances.dtype,
    )
    if width is None:
        width = (upper_bound - lower_bound) / num_bins
    return gaussian_radial_basis(distances, centers, width)
