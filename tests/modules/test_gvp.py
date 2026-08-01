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

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.gvp import Gvp, GvpDropout, GvpLayerNorm


@pytest.mark.parametrize("probability", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("training", [False, True])
def test_dropout_whole_vectors_endpoints_and_gradients(probability: float, training: bool) -> None:
    module = GvpDropout(probability).train(training)
    scalar = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    vector = torch.randn(3, 5, 3, dtype=torch.float64, requires_grad=True)
    scalar_output, vector_output = module((scalar, vector))
    assert torch.isfinite(scalar_output).all()
    assert torch.isfinite(vector_output).all()
    if not training or probability == 0:
        torch.testing.assert_close(scalar_output, scalar, rtol=0, atol=0)
        torch.testing.assert_close(vector_output, vector, rtol=0, atol=0)
    elif probability == 1:
        torch.testing.assert_close(scalar_output, torch.zeros_like(scalar), rtol=0, atol=0)
        torch.testing.assert_close(vector_output, torch.zeros_like(vector), rtol=0, atol=0)
    else:
        factor = vector_output / vector
        torch.testing.assert_close(factor, factor[..., :1].expand_as(factor), rtol=1e-12, atol=1e-12)
    scalar_gradient, vector_gradient = torch.autograd.grad(scalar_output.sum() + vector_output.sum(), (scalar, vector))
    assert torch.isfinite(scalar_gradient).all()
    assert torch.isfinite(vector_gradient).all()
    if training and probability == 1:
        torch.testing.assert_close(scalar_gradient, torch.zeros_like(scalar), rtol=0, atol=0)
        torch.testing.assert_close(vector_gradient, torch.zeros_like(vector), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("operation", ["perceptron", "normalization", "dropout"])
def test_rotation_equivariance_and_input_gradients(operation: str, dtype: torch.dtype) -> None:
    scalar = torch.randn(2, 4, dtype=dtype, requires_grad=True)
    vector = torch.randn(2, 3, 3, dtype=dtype, requires_grad=True)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    rotation = rotation.to(dtype=dtype)
    if operation == "perceptron":
        module = Gvp((4, 3), (5, 2)).to(dtype=dtype)
    elif operation == "normalization":
        module = GvpLayerNorm((4, 3)).to(dtype=dtype)
    else:
        module = GvpDropout(0.4).to(dtype=dtype).train()
    torch.manual_seed(42)
    scalar_output, vector_output = module((scalar, vector))
    torch.manual_seed(42)
    rotated_scalar, rotated_vector = module((scalar, vector @ rotation))
    torch.testing.assert_close(rotated_scalar, scalar_output, **_tolerance(dtype))
    torch.testing.assert_close(rotated_vector, vector_output @ rotation, **_tolerance(dtype))
    scalar_cotangent = torch.randn_like(scalar_output)
    vector_cotangent = torch.randn_like(vector_output)
    original_gradients = torch.autograd.grad(
        (scalar_output * scalar_cotangent).sum() + (vector_output * vector_cotangent).sum(), (scalar, vector)
    )
    rotated_gradients = torch.autograd.grad(
        (rotated_scalar * scalar_cotangent).sum() + (rotated_vector * (vector_cotangent @ rotation)).sum(),
        (scalar, vector),
    )
    for actual, expected in zip(rotated_gradients, original_gradients, strict=True):
        torch.testing.assert_close(actual, expected, **_tolerance(dtype))


_DTYPES = [torch.float64, torch.float32, torch.float16, torch.bfloat16]


def _tolerance(dtype: torch.dtype) -> dict[str, float]:
    return {
        torch.float64: {"rtol": 1e-11, "atol": 1e-11},
        torch.float32: {"rtol": 1e-5, "atol": 1e-6},
        torch.float16: {"rtol": 6e-3, "atol": 6e-3},
        torch.bfloat16: {"rtol": 6e-2, "atol": 6e-2},
    }[dtype]


@pytest.mark.parametrize("dtype", _DTYPES)
def test_perceptron_complete_equation_and_all_gradients(dtype: torch.dtype) -> None:
    module = Gvp((4, 3), (5, 2)).to(dtype=dtype)
    scalar = torch.randn(2, 4, dtype=dtype, requires_grad=True)
    vector = torch.randn(2, 3, 3, dtype=dtype)
    vector[0] = 0
    vector.requires_grad_()
    reference_scalar = scalar.detach().double().requires_grad_()
    reference_vector = vector.detach().double().requires_grad_()
    parameters = {name: value.detach().double().requires_grad_() for name, value in module.named_parameters()}
    hidden_vector = torch.einsum("bvc,hv->bhc", reference_vector, parameters["hidden_vector_projection.weight"])
    norms = hidden_vector.square().sum(-1).clamp_min(1e-8).sqrt()
    unactivated_scalar = F.linear(
        torch.cat((reference_scalar, norms), dim=-1),
        parameters["scalar_projection.weight"],
        parameters["scalar_projection.bias"],
    )
    expected_scalar = F.silu(unactivated_scalar)
    expected_vector = torch.einsum("bhc,vh->bvc", hidden_vector, parameters["output_vector_projection.weight"])
    expected_vector = expected_vector * F.linear(
        unactivated_scalar, parameters["vector_gate_projection.weight"], parameters["vector_gate_projection.bias"]
    ).sigmoid().unsqueeze(-1)
    actual_scalar, actual_vector = module((scalar, vector))
    torch.testing.assert_close(actual_scalar.double(), expected_scalar, **_tolerance(dtype))
    torch.testing.assert_close(actual_vector.double(), expected_vector, **_tolerance(dtype))
    scalar_cotangent = torch.randn_like(actual_scalar)
    vector_cotangent = torch.randn_like(actual_vector)
    actual_gradients = torch.autograd.grad(
        (actual_scalar * scalar_cotangent).sum() + (actual_vector * vector_cotangent).sum(),
        [scalar, vector, *module.parameters()],
    )
    expected_gradients = torch.autograd.grad(
        (expected_scalar * scalar_cotangent.double()).sum() + (expected_vector * vector_cotangent.double()).sum(),
        [reference_scalar, reference_vector, *parameters.values()],
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual.double(), expected, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("drop_all", [False, True])
def test_vector_rms_floor_before_mean_zero_large_and_all_gradients(dtype: torch.dtype, drop_all: bool) -> None:
    module = GvpLayerNorm((4, 3)).to(dtype=dtype)
    scalar = torch.randn(3, 4, dtype=dtype, requires_grad=True)
    vector = torch.tensor(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [2e-4, 0, 0], [4e-4, 0, 0]],
            [[500, 500, 500], [200, 100, 50], [-300, 400, 100]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    actual_input = GvpDropout(1.0).train()((scalar, vector)) if drop_all else (scalar, vector)
    actual_scalar, actual_vector = module(actual_input)
    assert torch.isfinite(actual_scalar).all() and torch.isfinite(actual_vector).all()
    reference_scalar = scalar.detach().double().requires_grad_()
    reference_vector = vector.detach().double().requires_grad_()
    parameters = {name: value.detach().double().requires_grad_() for name, value in module.named_parameters()}
    norm_scalar = reference_scalar * 0 if drop_all else reference_scalar
    norm_vector = reference_vector * 0 if drop_all else reference_vector
    expected_scalar = F.layer_norm(
        norm_scalar, (4,), parameters["scalar_norm.weight"], parameters["scalar_norm.bias"], module.scalar_norm.eps
    )
    # The lower bound applies independently to every vector before channel mean.
    expected_rms = norm_vector.square().sum(-1, keepdim=True).clamp_min(1e-8).mean(-2, keepdim=True).sqrt()
    expected_vector = norm_vector / expected_rms
    torch.testing.assert_close(actual_scalar.double(), expected_scalar, **_tolerance(dtype))
    torch.testing.assert_close(actual_vector.double(), expected_vector, **_tolerance(dtype))
    scalar_cotangent, vector_cotangent = torch.randn_like(actual_scalar), torch.randn_like(actual_vector)
    actual_gradients = torch.autograd.grad(
        (actual_scalar * scalar_cotangent).sum() + (actual_vector * vector_cotangent).sum(),
        [scalar, vector, *module.parameters()],
    )
    expected_gradients = torch.autograd.grad(
        (expected_scalar * scalar_cotangent.double()).sum() + (expected_vector * vector_cotangent.double()).sum(),
        [reference_scalar, reference_vector, *parameters.values()],
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), expected, **_tolerance(dtype))
