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
Tests for equivariant and CSR operators.
"""

import math

import pytest
import torch

from ofoldx.modules.backend import Se3Basis, SegmentSoftmaxCsr, SegmentSumCsr, WeightedTensorProduct


def test_segment_sum_csr() -> None:
    values = torch.tensor(((1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)))
    row_offsets = torch.tensor((0, 2, 2, 4))

    output = SegmentSumCsr(backend="eager")(values, row_offsets, num_segments=4)

    expected = torch.tensor(((3.0, 30.0), (0.0, 0.0), (7.0, 70.0), (0.0, 0.0)))
    torch.testing.assert_close(output, expected)


def test_segment_softmax_csr() -> None:
    logits = torch.tensor((0.0, math.log(3.0), 2.0, 2.0))
    row_offsets = torch.tensor((0, 2, 4))

    output = SegmentSoftmaxCsr(backend="eager")(logits, row_offsets)

    torch.testing.assert_close(output, torch.tensor((0.25, 0.75, 0.5, 0.5)))


@pytest.mark.parametrize("backend", ["eager", "auto"])
def test_weighted_tensor_product(backend: str) -> None:
    features = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    basis = torch.randn(2, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    radial_weights = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)

    output = WeightedTensorProduct(backend=backend)(features, basis, radial_weights)
    projected = torch.einsum("eci,eifo->ecfo", features, basis).reshape(2, 12, 3)
    expected = torch.matmul(radial_weights, projected)

    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (features, basis, radial_weights)
    )


def test_se3_basis() -> None:
    relative_positions = torch.tensor(
        ((1.0, 2.0, 3.0), (-2.0, 1.0, 0.5)),
        dtype=torch.float64,
        requires_grad=True,
    )

    degree_zero = Se3Basis(0, backend="eager")(relative_positions)
    assert {key: value.shape for key, value in degree_zero.items()} == {
        "0,0": (2, 1, 1, 1),
        "out0_fused": (2, 1, 1, 1),
        "in0_fused": (2, 1, 1, 1),
    }

    degree_one = Se3Basis(1, backend="eager")(relative_positions)
    assert {key: value.shape for key, value in degree_one.items()} == {
        "0,0": (2, 1, 1, 1),
        "0,1": (2, 1, 1, 3),
        "1,0": (2, 3, 1, 1),
        "1,1": (2, 3, 3, 3),
        "out0_fused": (2, 4, 2, 1),
        "out1_fused": (2, 4, 4, 3),
        "in0_fused": (2, 1, 2, 4),
        "in1_fused": (2, 3, 4, 4),
    }
    assert all(torch.isfinite(value).all() for value in (*degree_zero.values(), *degree_one.values()))

    degree_one["0,1"].sum().backward()
    assert relative_positions.grad is not None
    assert torch.isfinite(relative_positions.grad).all()
