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

r"""Logical batch and sample-axis tensor helpers.

Nested sample regrouping is expressed through DanLing's public packed interface
(``concat``, ``packed_offsets``, ``ragged_level_offsets``, ``element_sizes``,
``packed_dim_order``) and its ``NestedTensor.from_packed`` factory rather than
DanLing's reshaping methods: ``unflatten`` at the batch dimension is not
supported; ``flatten(0, 1)`` regroups ragged sample and atom axes correctly in
eager and FakeTensor execution but fails under fullgraph dynamic compilation
(nested shape metadata unavailable), and for a static sample axis it returns a
padded dense ``Tensor``, not a nested one.
"""

from __future__ import annotations

import torch
from danling.tensors import NestedTensor
from torch import Tensor


def insert_sample_axis(input: Tensor, num_samples: int, *, dim: int = -2) -> Tensor:
    r"""Insert and expand one sample axis of a tensor."""

    expanded = input.unsqueeze(dim)
    sizes = [-1] * expanded.dim()
    sizes[dim] = num_samples
    return expanded.expand(*sizes)


def repeat_batch(input: Tensor, num_repeats: int) -> Tensor:
    r"""Repeat complete logical batch elements in interleaved order.

    Dense tensors retain standard ``repeat_interleave(..., dim=0)`` semantics.
    DanLing ``NestedTensor`` inputs repeat packed sample segments and their
    tensor-backed topology without padding or per-element Python dispatch.
    """
    if num_repeats == 1:
        return input
    if isinstance(input, NestedTensor):
        return input.repeat_batch(num_repeats)
    return input.repeat_interleave(num_repeats, dim=0)


def unflatten_batch(input: Tensor, sizes: tuple[int, int]) -> Tensor:
    r"""Group interleaved samples from ``(B*S, N, ...)`` into ``(B, S, N, ...)``.

    Dense tensors use ordinary ``unflatten(0, sizes)``. A nested atom axis keeps
    its packed values and acquires two explicit ragged axes: the sample axis
    and the atom axis. Samples belonging to one batch element must share their
    atom count, while different batch elements may have different counts.
    """
    if not isinstance(input, NestedTensor):
        return input.unflatten(0, sizes)
    batch_size, num_samples = sizes
    if batch_size < 0 or num_samples <= 0 or batch_size * num_samples != input.shape[0]:
        raise ValueError("sizes must contain a nonnegative batch size and positive sample count matching the batch")
    if not input.batch_first or input.ragged_dims != (0,):
        raise ValueError("unflatten_batch requires a batch-first nested tensor with one leading ragged axis")

    element_sizes = input.element_sizes()
    grouped_sizes = element_sizes.reshape(batch_size, num_samples, element_sizes.shape[1])
    torch._assert_async(
        torch.all(grouped_sizes == grouped_sizes[:, :1]),
        "samples belonging to one batch element must have identical element sizes",
    )
    atom_offsets = input.packed_offsets()
    sample_offsets = torch.arange(batch_size + 1, dtype=atom_offsets.dtype) * num_samples
    shape = torch.cat((element_sizes.new_full((batch_size, 1), num_samples), element_sizes[::num_samples]), dim=1)
    return _from_packed(
        input,
        input.concat,
        atom_offsets[::num_samples],
        shape,
        permutation=(0, *(dim + 1 for dim in input.packed_dim_order)),
        ragged_dims=(0, 1),
        outer_size=(batch_size, num_samples, *input.shape[1:]),
        ragged_offsets=(sample_offsets, atom_offsets),
    )


def flatten_batch(input: Tensor) -> Tensor:
    r"""Flatten ``(B, S, N, ...)`` into interleaved ``(B*S, N, ...)`` samples.

    This reverses :func:`unflatten_batch` without copying nested packed values
    or introducing padding, including when all atom counts happen to agree.
    An externally supplied nested tensor may keep the sample axis static
    (``ragged_dims=(1,)``); its packed values are reordered into sample order.
    """
    if not isinstance(input, NestedTensor):
        return input.flatten(0, 1)
    if not input.batch_first or input.ragged_dims not in ((0, 1), (1,)):
        raise ValueError("flatten_batch requires a batch-first nested tensor with leading sample and atom axes")
    batch_size, num_samples = input.shape[:2]
    if num_samples <= 0:
        raise ValueError("every batch element must have the same positive sample count")
    element_sizes = input.element_sizes()
    torch._assert_async(
        torch.all(element_sizes[:, 0] == num_samples),
        "every batch element must have the same positive sample count",
    )
    if input.ragged_dims == (0, 1):
        atom_offsets = input.ragged_level_offsets(1)
        values = input.concat
    else:
        # Packed storage is atom-major with the static sample axis among the
        # static dimensions; gather it into sample-major order per element.
        offsets = input.packed_offsets()
        lengths = (offsets[1:] - offsets[:-1]).repeat_interleave(num_samples)
        atom_offsets = torch.cat((offsets.new_zeros(1), lengths.cumsum(0)))
        device = input.concat.device
        flat_batch = torch.repeat_interleave(
            torch.arange(batch_size * num_samples, device=device),
            lengths.to(device=device),
            output_size=input.concat.shape[0] * num_samples,
        )
        local_atom = torch.arange(flat_batch.numel(), device=device) - atom_offsets.to(device=device)[flat_batch]
        source_atom = offsets.to(device=device)[flat_batch // num_samples] + local_atom
        sample_dim = 1 + input.packed_dim_order[1:].index(0)
        values = input.concat.movedim(sample_dim, 1)[source_atom, flat_batch % num_samples]
    return _from_packed(
        input,
        values,
        atom_offsets,
        element_sizes[:, 1:].repeat_interleave(num_samples, dim=0),
        permutation=tuple(dim - 1 for dim in input.packed_dim_order if dim != 0),
        ragged_dims=(0,),
        outer_size=(batch_size * num_samples, *input.shape[2:]),
        ragged_offsets=(atom_offsets,),
    )


def _from_packed(
    reference: NestedTensor,
    values: Tensor,
    offsets: Tensor,
    shape: Tensor,
    *,
    permutation: tuple[int, ...],
    ragged_dims: tuple[int, ...],
    outer_size: tuple[int, ...],
    ragged_offsets: tuple[Tensor, ...],
) -> NestedTensor:
    return type(reference).from_packed(
        values,
        offsets,
        shape,
        permutation=permutation,
        ragged_dims=ragged_dims,
        batch_first=True,
        padding_value=reference.padding_value,
        mask_value=reference.mask_value,
        outer_size=outer_size,
        ragged_offsets=ragged_offsets,
        validate=False,
    )
