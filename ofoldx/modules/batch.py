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

r"""Logical batch and sample-axis tensor helpers."""

from __future__ import annotations

from torch import Tensor


def insert_sample_axis(input: Tensor, num_samples: int, *, dim: int = -2) -> Tensor:
    r"""Insert and expand one sample axis of a tensor."""

    expanded = input.unsqueeze(dim)
    sizes = [-1] * expanded.dim()
    sizes[dim] = num_samples
    return expanded.expand(*sizes)


def repeat_batch(input: Tensor, num_repeats: int) -> Tensor:
    r"""Repeat complete logical batch elements in interleaved order."""
    return input.repeat_interleave(num_repeats, dim=0)


def unflatten_batch(input: Tensor, sizes: tuple[int, int]) -> Tensor:
    r"""Group interleaved samples from ``(B*S, N, ...)`` into ``(B, S, N, ...)``."""
    return input.unflatten(0, sizes)


def flatten_batch(input: Tensor) -> Tensor:
    r"""Flatten ``(B, S, N, ...)`` into interleaved ``(B*S, N, ...)`` samples."""
    return input.flatten(0, 1)
