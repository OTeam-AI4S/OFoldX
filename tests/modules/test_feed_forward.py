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

from ofoldx.models.alphafold2.model import AlphaFold2SidechainBlock
from ofoldx.models.rfdiffusion.model import RfDiffusionTorsionBlock
from ofoldx.modules.feed_forward import FeedForward, SwiGluFeedForward


@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("depth", [0, -1])
def test_nonlinear_feed_forward_requires_positive_hidden_depth(activation: str, depth: int) -> None:
    with pytest.raises(ValueError, match="num_hidden_layers must be positive"):
        FeedForward(3, 5, 2, num_hidden_layers=depth, activation=activation)


@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("probability", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("training", [False, True])
def test_affine_activation_depth_and_all_gradients(
    activation: str, depth: int, bias: bool, probability: float, training: bool
) -> None:
    module = (
        FeedForward(3, 5, 2, num_hidden_layers=depth, bias=bias, dropout=probability, activation=activation)
        .double()
        .train(training)
    )
    actual_input = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_()
    references = {name: parameter.detach().clone().requires_grad_() for name, parameter in module.named_parameters()}
    hidden = expected_input
    stages = ["input_projection", *(f"intermediate_projections.{index}" for index in range(depth - 1))]
    torch.manual_seed(31)
    actual = module(actual_input)
    torch.manual_seed(31)
    for stage in stages:
        projected = F.linear(hidden, references[f"{stage}.weight"], references.get(f"{stage}.bias"))
        multiplier = F.dropout(torch.ones_like(projected), p=probability, training=training)
        # ReLU's independent reference keeps the former post-activation dropout
        # equation. GELU retains its pre-activation dropout equation.
        hidden = multiplier * F.relu(projected) if activation == "relu" else F.gelu(multiplier * projected)
    expected = F.linear(hidden, references["output_projection.weight"], references.get("output_projection.bias"))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, [actual_input, *module.parameters()], cotangent)
    expected_gradients = torch.autograd.grad(expected, [expected_input, *references.values()], cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("block_type", [AlphaFold2SidechainBlock, RfDiffusionTorsionBlock])
def test_preactivation_keeps_identity_residual_and_leaf_gradients(block_type: type) -> None:
    module = block_type(3).double()
    hidden = torch.tensor([[-2.0, 1.0, -3.0]], dtype=torch.float64, requires_grad=True)
    original = hidden.detach().clone()
    actual = module(hidden)
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    gradient = torch.autograd.grad(actual.sum(), hidden)[0]
    torch.testing.assert_close(gradient, torch.ones_like(hidden), rtol=0, atol=0)


class TestSwiGluFeedForward:
    def test_forward_and_gradients(self) -> None:
        module = SwiGluFeedForward(3, 5, 2, bias=True)
        hidden = torch.randn(4, 3, requires_grad=True)

        actual = module(hidden)
        expected = F.linear(
            F.silu(F.linear(hidden, module.gate_projection.weight, module.gate_projection.bias))
            * F.linear(hidden, module.value_projection.weight, module.value_projection.bias),
            module.output_projection.weight,
            module.output_projection.bias,
        )
        torch.testing.assert_close(actual, expected)

        actual.sum().backward()
        assert hidden.grad is not None


@pytest.mark.parametrize("norm_type", [torch.nn.BatchNorm1d, torch.nn.LayerNorm])
@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize(("hidden_bias", "output_bias"), [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("probability", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("training", [False, True])
def test_normalized_hidden_stages_output_bias_gradients_and_running_state(
    norm_type: type, depth: int, hidden_bias: bool, output_bias: bool, probability: float, training: bool
) -> None:
    module = (
        FeedForward(
            3,
            5,
            2,
            num_hidden_layers=depth,
            bias=hidden_bias,
            output_bias=output_bias,
            norm=norm_type,
            dropout=probability,
            activation="relu",
        )
        .double()
        .train(training)
    )
    actual_input = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_()
    references = {name: value.detach().clone().requires_grad_() for name, value in module.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in module.named_buffers()}
    torch.manual_seed(37)
    actual = module(actual_input)
    torch.manual_seed(37)
    expected = expected_input
    stages = [
        ("input_projection", "input_norm"),
        *((f"intermediate_projections.{index}", f"intermediate_norms.{index}") for index in range(depth - 1)),
    ]
    for projection, norm in stages:
        expected = F.linear(expected, references[f"{projection}.weight"], references.get(f"{projection}.bias"))
        if norm_type is torch.nn.BatchNorm1d:
            if training:
                mean = expected.mean(dim=0)
                variance = (expected - mean).square().mean(dim=0)
                buffers[f"{norm}.running_mean"] = 0.9 * buffers[f"{norm}.running_mean"] + 0.1 * mean.detach()
                buffers[f"{norm}.running_var"] = 0.9 * buffers[f"{norm}.running_var"] + 0.1 * variance.detach() * (
                    8 / 7
                )
                buffers[f"{norm}.num_batches_tracked"] += 1
            else:
                mean, variance = buffers[f"{norm}.running_mean"], buffers[f"{norm}.running_var"]
        else:
            mean = expected.mean(dim=-1, keepdim=True)
            variance = (expected - mean).square().mean(dim=-1, keepdim=True)
        expected = (expected - mean) / (variance + 1e-5).sqrt()
        expected = expected * references[f"{norm}.weight"] + references[f"{norm}.bias"]
        multiplier = F.dropout(torch.ones_like(expected), p=probability, training=training)
        expected = multiplier * F.relu(expected)
    expected = F.linear(expected, references["output_projection.weight"], references.get("output_projection.bias"))
    torch.testing.assert_close(actual, expected, rtol=5e-10, atol=5e-11)
    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (actual_input, *module.parameters()), cotangent)
    expected_gradients = torch.autograd.grad(expected, (expected_input, *references.values()), cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-10, atol=5e-11)
    for name, value in module.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=5e-10, atol=5e-11)


@pytest.mark.parametrize(("hidden_bias", "output_bias"), [(False, True), (True, False)])
def test_identity_normalization_supports_independent_output_bias(hidden_bias: bool, output_bias: bool) -> None:
    module = FeedForward(3, 5, 2, bias=hidden_bias, output_bias=output_bias).double()
    hidden = torch.randn(4, 3, dtype=torch.float64)
    expected = F.linear(
        F.relu(F.linear(hidden, module.input_projection.weight, module.input_projection.bias)),
        module.output_projection.weight,
        module.output_projection.bias,
    )
    torch.testing.assert_close(module(hidden), expected, rtol=1e-12, atol=1e-12)
